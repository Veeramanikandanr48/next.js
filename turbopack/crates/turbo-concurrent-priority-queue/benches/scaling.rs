//! Scaling benchmarks: how throughput changes with the number of threads, for
//! [`ConcurrentPriorityQueue`] compared with a `Mutex<BinaryHeap>` (strict priority order) and an
//! unbounded `ConcurrentQueue` (lock-free, but no priority order at all, so it is a throughput
//! reference rather than a drop-in alternative).
//!
//! Every benchmark iteration performs [`TOTAL_OPS`] queue operations in total, split evenly across
//! the threads, so the time per iteration directly shows how well an implementation scales: perfect
//! scaling halves it when the thread count doubles. Benchmark ids are
//! `turbo_concurrent_priority_queue_scaling/<workload>/<mix>/<impl>/<threads>`.
//!
//! Workloads:
//! - `mixed`: the queue is prefilled with [`PREFILL`] items; every thread repeatedly pushes one
//!   item and pops one (steady state, the queue never runs empty);
//! - `drain`: the queue is prefilled with [`TOTAL_OPS`] items (not timed); all threads pop until it
//!   is empty, each exactly its share;
//! - `fill`: all threads only push; the queue is drained afterwards (not timed).
//!
//! Priority mixes:
//! - `single`: every item has priority 0;
//! - `skewed`: like a build: most items have the lowest priority, a few have one of three higher
//!   ones.
//!
//! The worker threads are spawned once per Criterion sample; each iteration releases them with a
//! start barrier and waits for them with an end barrier. Only the time between the two (including
//! the barrier hand-off) is measured; prefilling, draining and checking happen outside of it. Every
//! iteration checks its item counts, so a broken queue makes the benchmark panic instead of
//! reporting a meaningless number.

use std::{
    collections::BinaryHeap,
    hint::black_box,
    sync::Barrier,
    thread,
    time::{Duration, Instant},
};

use concurrent_queue::ConcurrentQueue;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parking_lot::Mutex;
use turbo_concurrent_priority_queue::ConcurrentPriorityQueue;

/// Queue operations per iteration (all threads together). Divisible by `2 * 32`, so every thread
/// count gets an exact share, and `mixed` an exact number of push + pop pairs.
const TOTAL_OPS: usize = 1 << 16;
/// Items in the queue before a `mixed` iteration starts.
const PREFILL: usize = 4096;
const THREAD_COUNTS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// The operations the benchmarks need, implemented by every compared queue.
trait BenchQueue: Sync {
    const NAME: &'static str;
    fn new() -> Self;
    fn push(&self, priority: u32, item: u64);
    fn pop(&self) -> Option<u64>;
}

impl BenchQueue for ConcurrentPriorityQueue<u32, u64> {
    const NAME: &'static str = "sharded";

    fn new() -> Self {
        ConcurrentPriorityQueue::new()
    }

    fn push(&self, priority: u32, item: u64) {
        ConcurrentPriorityQueue::push(self, priority, item);
    }

    fn pop(&self) -> Option<u64> {
        ConcurrentPriorityQueue::pop(self).map(|(_, item)| item)
    }
}

struct MutexHeap(Mutex<BinaryHeap<(u32, u64)>>);

impl BenchQueue for MutexHeap {
    const NAME: &'static str = "mutex_heap";

    fn new() -> Self {
        MutexHeap(Mutex::new(BinaryHeap::new()))
    }

    fn push(&self, priority: u32, item: u64) {
        self.0.lock().push((priority, item));
    }

    fn pop(&self) -> Option<u64> {
        self.0.lock().pop().map(|(_, item)| item)
    }
}

struct Unordered(ConcurrentQueue<(u32, u64)>);

impl BenchQueue for Unordered {
    const NAME: &'static str = "concurrent_queue";

    fn new() -> Self {
        Unordered(ConcurrentQueue::unbounded())
    }

    fn push(&self, priority: u32, item: u64) {
        self.0
            .push((priority, item))
            .expect("an unbounded queue that is never closed accepts every push");
    }

    fn pop(&self) -> Option<u64> {
        self.0.pop().ok().map(|(_, item)| item)
    }
}

#[derive(Clone, Copy)]
enum Workload {
    Mixed,
    Drain,
    Fill,
}

impl Workload {
    const ALL: [Workload; 3] = [Workload::Mixed, Workload::Drain, Workload::Fill];

    fn name(self) -> &'static str {
        match self {
            Workload::Mixed => "mixed",
            Workload::Drain => "drain",
            Workload::Fill => "fill",
        }
    }
}

#[derive(Clone, Copy)]
enum Mix {
    Single,
    Skewed,
}

impl Mix {
    const ALL: [Mix; 2] = [Mix::Single, Mix::Skewed];

    fn name(self) -> &'static str {
        match self {
            Mix::Single => "single",
            Mix::Skewed => "skewed",
        }
    }

    /// The priority of the `index`-th item pushed by thread `thread` (deterministic).
    fn priority(self, thread: usize, index: usize) -> u32 {
        match self {
            Mix::Single => 0,
            Mix::Skewed => {
                let hash =
                    ((thread as u64) << 32 | index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                match hash >> 60 {
                    0..=11 => 0,
                    12 | 13 => 1,
                    14 => 2,
                    _ => 3,
                }
            }
        }
    }
}

/// A unique item value for the `index`-th item pushed by thread `thread`.
fn item(thread: usize, index: usize) -> u64 {
    (thread as u64) << 32 | index as u64
}

/// Thread index used for items pushed outside of the timed region (prefill).
const SETUP_THREAD: usize = u32::MAX as usize;

/// The work one thread does in one iteration.
fn work<Q: BenchQueue>(queue: &Q, workload: Workload, mix: Mix, thread: usize, threads: usize) {
    let share = TOTAL_OPS / threads;
    match workload {
        Workload::Mixed => {
            for index in 0..share / 2 {
                queue.push(mix.priority(thread, index), item(thread, index));
                // This thread's push precedes its pop, so the queue holds at least `PREFILL` items.
                black_box(queue.pop().expect("mixed: the queue never runs empty"));
            }
        }
        Workload::Drain => {
            for _ in 0..share {
                black_box(
                    queue
                        .pop()
                        .expect("drain: every thread pops exactly its share"),
                );
            }
        }
        Workload::Fill => {
            for index in 0..share {
                queue.push(mix.priority(thread, index), item(thread, index));
            }
        }
    }
}

fn prefill<Q: BenchQueue>(queue: &Q, mix: Mix, count: usize) {
    for index in 0..count {
        queue.push(mix.priority(SETUP_THREAD, index), item(SETUP_THREAD, index));
    }
}

fn drain_count<Q: BenchQueue>(queue: &Q) -> usize {
    std::iter::from_fn(|| queue.pop()).count()
}

/// Runs `iters` iterations with `threads` worker threads and returns the measured time.
fn run<Q: BenchQueue>(workload: Workload, mix: Mix, threads: usize, iters: u64) -> Duration {
    let queue = Q::new();
    let start = Barrier::new(threads + 1);
    let end = Barrier::new(threads + 1);
    thread::scope(|scope| {
        for thread in 0..threads {
            let (queue, start, end) = (&queue, &start, &end);
            scope.spawn(move || {
                for _ in 0..iters {
                    start.wait();
                    work(queue, workload, mix, thread, threads);
                    end.wait();
                }
            });
        }
        let mut elapsed = Duration::ZERO;
        for _ in 0..iters {
            // Workers are waiting at the start barrier, so the queue is ours.
            match workload {
                Workload::Mixed => prefill(&queue, mix, PREFILL),
                Workload::Drain => prefill(&queue, mix, TOTAL_OPS),
                Workload::Fill => {}
            }
            let begin = Instant::now();
            start.wait();
            end.wait();
            elapsed += begin.elapsed();
            let left = drain_count(&queue);
            let expected = match workload {
                Workload::Mixed => PREFILL,
                Workload::Drain => 0,
                Workload::Fill => TOTAL_OPS,
            };
            assert_eq!(
                left,
                expected,
                "{}/{}/{}/{threads}: items left in the queue",
                workload.name(),
                mix.name(),
                Q::NAME
            );
        }
        elapsed
    })
}

fn bench_impl<Q: BenchQueue>(c: &mut Criterion, thread_counts: &[usize]) {
    let mut group = c.benchmark_group("turbo_concurrent_priority_queue_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(TOTAL_OPS as u64));
    for workload in Workload::ALL {
        for mix in Mix::ALL {
            for &threads in thread_counts {
                let id = BenchmarkId::new(
                    format!("{}/{}/{}", workload.name(), mix.name(), Q::NAME),
                    threads,
                );
                group.bench_function(id, |b| {
                    b.iter_custom(|iters| run::<Q>(workload, mix, threads, iters))
                });
            }
        }
    }
    group.finish();
}

fn scaling(c: &mut Criterion) {
    let available = thread::available_parallelism().map_or(1, |n| n.get());
    let thread_counts: Vec<usize> = THREAD_COUNTS
        .into_iter()
        .filter(|&threads| threads == 1 || threads <= available)
        .collect();
    bench_impl::<ConcurrentPriorityQueue<u32, u64>>(c, &thread_counts);
    bench_impl::<MutexHeap>(c, &thread_counts);
    bench_impl::<Unordered>(c, &thread_counts);
}

criterion_group!(benches, scaling);
criterion_main!(benches);

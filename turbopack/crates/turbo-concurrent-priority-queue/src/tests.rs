use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, AtomicU64},
    },
    time::Duration,
};

use super::*;

// The tests use plain `u32` priorities: a larger value is a higher priority and its own rank.

type Queue<T> = ConcurrentPriorityQueue<u32, T>;

/// Runs `f` with this thread's shard forced to `index`, restoring the previous shard after.
fn on_shard<R>(index: usize, f: impl FnOnce() -> R) -> R {
    let previous = thread_shard();
    set_thread_shard(index);
    let result = f();
    set_thread_shard(previous);
    result
}

fn state<T>(q: &Queue<T>) -> u64 {
    q.state.load(Ordering::Acquire)
}

fn shard_len<T>(q: &Queue<T>, index: usize) -> usize {
    q.shards[index].lock().buckets.values().map(Vec::len).sum()
}

fn all_shards_empty<T>(q: &Queue<T>) -> bool {
    q.shards.iter().all(|shard| shard.lock().is_empty())
}

fn drain<T>(q: &Queue<T>) -> Vec<T> {
    std::iter::from_fn(|| q.pop().map(|(_, item)| item)).collect()
}

/// Drives a search to its end by hand.
fn run_search<T>(q: &Queue<T>) {
    loop {
        let state = state(q);
        if !is_searching(state) {
            return;
        }
        q.help_search(state);
    }
}

/// From a single thread, priorities pop highest first; within a priority a single shard is LIFO.
#[test]
fn pops_in_exact_priority_order() {
    let q: Queue<&str> = Queue::new();
    assert!(q.is_empty());
    for (priority, item) in [
        (0, "low-a"),
        (3, "p3"),
        (9, "p9-a"),
        (0, "low-b"),
        (7, "p7"),
        (5, "p5"),
        (9, "p9-b"),
    ] {
        q.push(priority, item);
    }
    assert!(!q.is_empty());
    assert_eq!(
        drain(&q),
        vec!["p9-b", "p9-a", "p7", "p5", "p3", "low-b", "low-a"]
    );
    assert!(q.is_empty());
    assert!(all_shards_empty(&q));
}

#[test]
fn push_many_queues_all_items() {
    let q: Queue<u32> = Queue::new();
    q.push_many(std::iter::empty());
    assert!(q.is_empty());
    q.push_many((0..8).map(|item| (if item % 2 == 0 { 0 } else { item }, item)));
    assert_eq!(drain(&q), vec![7, 5, 3, 1, 6, 4, 2, 0]);
    assert!(q.is_empty());
}

/// Popping the last item of a rank above 0 leaves a stale `highest` without bits in the state
/// word (until the next pop searches); `is_empty` must still report the drained queue as empty.
#[test]
fn is_empty_after_draining_a_higher_rank() {
    let q: Queue<u32> = Queue::new();
    q.push(5, 5);
    assert!(!q.is_empty());
    assert_eq!(q.pop(), Some((5, 5)));
    assert!(q.is_empty());
    assert!(q.pop().is_none());
    assert!(q.is_empty());
}

/// A panicking iterator passed to `push_many` must not leave items in a shard that the state word
/// doesn't know about: every item that was queued stays poppable.
#[test]
fn push_many_panicking_iterator_loses_nothing() {
    let q: Queue<u32> = Queue::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        q.push_many((0..3).map(|i| {
            if i == 1 {
                panic!("iterator panic");
            }
            (7, i)
        }));
    }));
    assert!(result.is_err());
    let popped = drain(&q);
    assert!(
        all_shards_empty(&q),
        "items left in a shard but not poppable (popped {popped:?})"
    );
    assert!(q.is_empty());
}

/// Rank 0 is the lowest rank: draining it empties the queue by clearing the last shard bit,
/// without a search for a new highest priority.
#[test]
fn rank_zero_drain_reaches_empty_without_search() {
    let q: Queue<u32> = Queue::new();
    on_shard(0, || (0..10).for_each(|item| q.push(0, item)));
    on_shard(1, || (10..20).for_each(|item| q.push(0, item)));
    assert_eq!(state(&q), make_state(0, false, shard_bit(0) | shard_bit(1)));
    let mut popped = 0;
    on_shard(2, || {
        while q.pop().is_some() {
            popped += 1;
            assert!(!is_searching(state(&q)));
        }
    });
    assert_eq!(popped, 20);
    assert_eq!(state(&q), 0);
    assert!(all_shards_empty(&q));
}

/// A thread whose shard doesn't have the highest priority steals the newest half of a victim's
/// items at that priority, and the victim keeps the rest.
#[test]
fn steal_moves_half_of_the_highest_priority() {
    let q: Queue<u32> = Queue::new();
    on_shard(0, || {
        (0..10).for_each(|item| q.push(5, item));
        // A lower priority stays behind in the victim.
        q.push(1, 100);
    });
    let (_, item) = on_shard(1, || q.pop()).unwrap();
    // The newest item is popped from the stolen half (5..=9).
    assert_eq!(item, 9);
    assert_eq!(shard_len(&q, 1), 4, "5 stolen, 1 popped");
    assert_eq!(
        shard_len(&q, 0),
        6,
        "5 left at priority 5, plus the lower priority item"
    );
    assert_eq!(state(&q), make_state(5, false, shard_bit(0) | shard_bit(1)));

    // Popping the last items at priority 5 of shard 0 clears its bit.
    on_shard(0, || {
        for expected in [4, 3, 2, 1, 0] {
            assert_eq!(q.pop().unwrap().1, expected);
        }
    });
    assert_eq!(state(&q), make_state(5, false, shard_bit(1)));
    assert_eq!(on_shard(2, || drain(&q)), vec![8, 7, 6, 5, 100]);
    assert_eq!(state(&q), 0);
}

/// Once no shard has the highest priority, a search finds the next one across shards.
#[test]
fn search_finds_next_priority() {
    let q: Queue<u32> = Queue::new();
    on_shard(0, || q.push(3, 3));
    on_shard(1, || q.push(7, 7));
    on_shard(2, || q.push(5, 5));
    // The push of 7 reset the bits to its own shard; 5 is lower than 7 and wrote nothing.
    assert_eq!(state(&q), make_state(7, false, shard_bit(1)));
    assert_eq!(on_shard(3, || drain(&q)), vec![7, 5, 3]);
    assert_eq!(state(&q), 0);
    assert!(all_shards_empty(&q));
}

/// A search that only finds rank-0 items can't tell them apart from empty shards in `highest`, so
/// it ends with all bits set; the items are still popped, and lazy clearing then reaches the empty
/// state.
#[test]
fn search_with_only_rank_zero_items_ends_with_all_bits() {
    let q: Queue<u32> = Queue::new();
    on_shard(0, || q.push(0, 0));
    on_shard(1, || q.push(4, 4));
    assert_eq!(on_shard(1, || q.pop()).unwrap().1, 4);
    // Shard 1 cleared its bit: nobody has rank 4 any more.
    assert_eq!(state(&q), make_state(4, false, 0));
    q.state
        .compare_exchange(
            make_state(4, false, 0),
            SEARCHING,
            Ordering::AcqRel,
            Ordering::Relaxed,
        )
        .unwrap();
    run_search(&q);
    assert_eq!(state(&q), make_state(0, false, ALL_BITS));
    assert_eq!(on_shard(5, || q.pop()).unwrap().1, 0);
    assert!(on_shard(5, || q.pop()).is_none());
    assert_eq!(state(&q), 0);
}

/// A search over empty shards ends with all bits set, which lazy clearing turns into the empty
/// state, so `pop` returns `None`.
#[test]
fn search_over_empty_shards_ends_empty() {
    let q: Queue<u32> = Queue::new();
    // A stale `highest` with no bits, as left behind by the last pop of a priority.
    q.state.store(make_state(4, false, 0), Ordering::Release);
    assert!(on_shard(3, || q.pop()).is_none());
    assert_eq!(state(&q), 0);
}

/// Pushes keep the state word up to date: a higher priority resets the bits to the pusher's shard,
/// an equal one adds its bit, a lower one writes nothing.
#[test]
fn push_updates_state_word() {
    let q: Queue<u32> = Queue::new();
    on_shard(0, || q.push(3, 0));
    on_shard(1, || q.push(3, 1));
    assert_eq!(state(&q), make_state(3, false, shard_bit(0) | shard_bit(1)));
    on_shard(2, || q.push(9, 2));
    assert_eq!(state(&q), make_state(9, false, shard_bit(2)));
    on_shard(3, || q.push(1, 3));
    assert_eq!(state(&q), make_state(9, false, shard_bit(2)));
    on_shard(4, || q.push_many([(9, 4), (2, 5)]));
    assert_eq!(state(&q), make_state(9, false, shard_bit(2) | shard_bit(4)));
}

/// A push during a search counts as a check of its shard, even if that shard was already checked,
/// so the search can't end below it.
#[test]
fn push_during_search_is_checked() {
    let q: Queue<u32> = Queue::new();
    on_shard(0, || q.push(2, 2));
    // Simulate a search that started and already checked the (empty) shard 5.
    q.state.store(SEARCHING, Ordering::Release);
    q.check_shard(5, &q.shards[5].lock());
    assert_eq!(state(&q), make_state(0, true, shard_bit(5)));
    on_shard(5, || q.push(8, 8));
    assert_eq!(state(&q), make_state(8, true, shard_bit(5)));
    run_search(&q);
    assert_eq!(state(&q), make_state(8, false, ALL_BITS));
    assert_eq!(on_shard(9, || drain(&q)), vec![8, 2]);
    assert_eq!(state(&q), 0);
}

/// Priorities of equal rank are ordered by the full priority within a shard.
#[test]
fn equal_rank_priorities_are_ordered_within_a_shard() {
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    struct Coarse(u32);
    impl QueuePriority for Coarse {
        fn rank(&self) -> u32 {
            self.0 / 10
        }
    }
    let q: ConcurrentPriorityQueue<Coarse, u32> = ConcurrentPriorityQueue::new();
    for p in [11, 15, 3, 19, 12] {
        q.push(Coarse(p), p);
    }
    assert_eq!(
        std::iter::from_fn(|| q.pop().map(|(_, item)| item)).collect::<Vec<_>>(),
        vec![19, 15, 12, 11, 3]
    );
}

/// A thread whose shard is locked by someone else moves to another shard instead of waiting.
#[test]
fn busy_shard_is_skipped() {
    let q: Arc<Queue<u32>> = Arc::new(Queue::new());
    let guard = q.shards[0].lock();
    let (sender, receiver) = std::sync::mpsc::channel();
    let worker = {
        let q = q.clone();
        std::thread::spawn(move || {
            set_thread_shard(0);
            q.push(1, 1);
            let popped = q.pop().map(|(_, item)| item);
            sender.send((popped, thread_shard())).unwrap();
        })
    };
    let result = receiver.recv_timeout(Duration::from_secs(5));
    drop(guard);
    worker.join().unwrap();
    let (popped, shard) = result.expect("push and pop must not wait for a busy shard");
    assert_eq!(popped, Some(1));
    assert_ne!(shard, 0, "the thread sticks to the shard it moved to");
}

/// Many threads push, pop, steal and search concurrently over many rounds. Checks that every item
/// is popped exactly once, and, in the quiescent phase at the end of each round (no pushes), the
/// ordering guarantee: no item of a higher priority is popped by a pop that starts after another
/// pop of a lower priority (or a `None`) has completed.
#[test]
fn stress_no_loss_no_duplicates_and_order() {
    const ROUNDS: u32 = 20;
    const PRODUCERS: u32 = 4;
    const CONSUMERS: u32 = 6;
    const PER_PRODUCER: u32 = 2_000;
    const PER_ROUND: u32 = PRODUCERS * PER_PRODUCER;

    /// Pops of the quiescent phase: `(start, end, priority or None)` in a global sequence.
    type QuiescentPop = (u64, u64, Option<u32>);

    let total = ROUNDS * PER_ROUND;
    let popped_count: Vec<AtomicU32> = (0..total).map(|_| AtomicU32::new(0)).collect();
    let q: Queue<u32> = Queue::new();
    let seq = AtomicU64::new(0);

    for round in 0..ROUNDS {
        let base = round * PER_ROUND;
        let producers_done = AtomicU32::new(0);
        let quiescent = AtomicBool::new(false);
        let quiescent_pops: Mutex<Vec<QuiescentPop>> = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for consumer in 0..CONSUMERS {
                let (q, popped_count, seq, quiescent, quiescent_pops) =
                    (&q, &popped_count, &seq, &quiescent, &quiescent_pops);
                scope.spawn(move || {
                    set_thread_shard((consumer as usize * 5) % SHARD_COUNT);
                    let mut local = Vec::new();
                    loop {
                        let in_quiescent = quiescent.load(Ordering::Acquire);
                        let start = seq.fetch_add(1, Ordering::SeqCst);
                        let popped = q.pop();
                        let end = seq.fetch_add(1, Ordering::SeqCst);
                        if let Some((_, id)) = &popped {
                            popped_count[*id as usize].fetch_add(1, Ordering::Relaxed);
                        }
                        if in_quiescent {
                            local.push((start, end, popped.as_ref().map(|(p, _)| *p)));
                            if popped.is_none() {
                                break;
                            }
                        }
                    }
                    quiescent_pops.lock().extend(local);
                });
            }
            for producer in 0..PRODUCERS {
                let (q, producers_done) = (&q, &producers_done);
                scope.spawn(move || {
                    set_thread_shard((producer as usize * 7 + 3) % SHARD_COUNT);
                    let mut x = 0x9E37_79B9u32 ^ (producer << 8) ^ round;
                    let mut batch = Vec::new();
                    for i in 0..PER_PRODUCER {
                        x ^= x << 13;
                        x ^= x >> 17;
                        x ^= x << 5;
                        let id = base + producer * PER_PRODUCER + i;
                        // Mostly a few priorities (like real builds), sometimes a spread.
                        let priority = match x % 8 {
                            0..=3 => 0,
                            4 | 5 => 3,
                            6 => 7,
                            _ => x % 64,
                        };
                        if x % 32 < 4 {
                            batch.push((priority, id));
                            if batch.len() == 8 {
                                q.push_many(std::mem::take(&mut batch));
                            }
                        } else {
                            q.push(priority, id);
                        }
                    }
                    q.push_many(batch);
                    producers_done.fetch_add(1, Ordering::AcqRel);
                });
            }
            while producers_done.load(Ordering::Acquire) < PRODUCERS {
                std::thread::yield_now();
            }
            quiescent.store(true, Ordering::Release);
        });

        assert!(q.pop().is_none());
        assert!(all_shards_empty(&q), "round {round}: items left behind");
        assert_eq!(state(&q), 0);

        // Ordering in the quiescent phase.
        let pops = quiescent_pops.into_inner();
        let mut by_start = pops.clone();
        by_start.sort_by_key(|&(start, _, _)| start);
        for &(_, end, priority) in &pops {
            for &(start, _, later) in by_start.iter().rev() {
                if start < end {
                    break;
                }
                if let Some(later) = later {
                    assert!(
                        priority.is_some_and(|priority| later <= priority),
                        "round {round}: a pop of priority {later} started after a pop of \
                         {priority:?} ended"
                    );
                }
            }
        }
    }

    for (id, count) in popped_count.iter().enumerate() {
        assert_eq!(count.load(Ordering::Relaxed), 1, "item {id}");
    }
}

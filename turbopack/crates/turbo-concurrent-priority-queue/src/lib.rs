//! A concurrent priority queue for many threads that push and pop a lot:
//! [`ConcurrentPriorityQueue`].
//!
//! The queue is a set of [`SHARD_COUNT`] mutex-protected shards plus one shared atomic state word.
//! A thread sticks to one shard (a thread-local index), so in the common case pushes and pops of
//! different threads take different, uncontended locks. Threads that run out of work at the
//! highest priority steal half of another shard's items at that priority.
//!
//! # Shards
//!
//! A shard maps each priority it contains to the items queued with it (a `Vec`), and keeps a heap
//! of these priorities (one heap entry per distinct priority, not per item). Items with the same
//! priority are popped in LIFO order *within a shard* (the newest first, which keeps recently
//! queued work hot); across shards there is no order among equal priorities.
//!
//! # The state word
//!
//! `state` is an `AtomicU64` made of:
//! - `highest` (bits 32..64): the [`QueuePriority::rank`] of the highest queued priority. While a
//!   search is in progress it is the maximum found so far (starting at `0`);
//! - `SEARCHING` (bit 31): the "find new priority" flag;
//! - 31 shard bits (bits 0..31), one per shard.
//!
//! Ranks are monotone but not strictly so (see [`QueuePriority`]): shards order by the full
//! priority, the state word only by rank. "A shard has `highest`" means the shard is not empty and
//! the top of its heap has `rank == highest` (an empty shard never has `highest`, not even `0`).
//!
//! ## Invariants
//!
//! - **I1 (no hidden higher work):** while `SEARCHING` is clear, no shard's top priority has a rank
//!   above `highest`. Pushes raise `highest` while still holding the shard lock, and a search only
//!   finishes after it looked at every shard (see below).
//! - **I2 (superset bits):** while `SEARCHING` is clear, every shard that has `highest` has its bit
//!   set. A bit may be set spuriously (the shard doesn't have `highest`); any thread that holds
//!   that shard's lock and sees it doesn't have `highest` clears it.
//! - **I3 (search):** while `SEARCHING` is set, a set bit means the shard was *checked*: its top
//!   rank at the time of the check (under its lock), and the top rank after every later push to it,
//!   has been max-ed into `highest`. Unset bits are shards that still need a check.
//! - **I4 (empty):** `state == 0` (not searching, `highest == 0`, no bits) means the queue is
//!   empty: by I2 no shard has rank `0`, and by I1 none has more. Rank `0` is the lowest rank, so
//!   draining it reaches the empty state directly by clearing the last bit, without a search. `pop`
//!   only returns `None` in this state.
//! - **I5 (locks):** a shard bit is only *cleared* by a thread holding that shard's lock (or by a
//!   transition that discards all bits: raising `highest`, starting a search). Items moved by a
//!   steal are never outside a locked shard. At most two shard locks are held at once, and the
//!   second one is only ever taken with `try_lock`, so shard locks can't deadlock. No other lock is
//!   taken while holding a shard lock.
//!
//! ## Transitions
//!
//! All transitions are CAS loops on `state` whose decision only depends on the current word and on
//! the state of shards whose lock the thread holds, so a word that changes and changes back (ABA)
//! can't make a decision wrong.
//!
//! - **Push** to shard `i` (holding its lock, after inserting) publishes the shard's new top rank
//!   `r`:
//!   - not searching, `r > highest`: `(r, only bit i)`. The old bits refer to a lower priority and
//!     are dropped (I2 holds for the new `highest`: only shard `i` can have it, by I1).
//!   - not searching, `r == highest`: set bit `i`.
//!   - not searching, `r < highest`: no write.
//!   - searching: the push is a check of shard `i`: max `r` into `highest` and set bit `i` (I3).
//! - **Pop** from shard `i` (holding its lock): re-read `state` under the lock and only pop if the
//!   shard has `highest` ("peek first"). If the shard no longer has `highest` afterwards, clear bit
//!   `i` (only if `highest` is unchanged and not searching).
//! - **Steal:** if shard `i` doesn't have `highest`, keep holding its lock and `try_lock` a shard
//!   `j` whose bit is set. If `j` has `highest`, move half (rounded up) of its items at that
//!   priority, the newest ones, to shard `i`, publish `i`, then clear bit `j` if `j` no longer has
//!   `highest`. If `j` doesn't have `highest`, clear its spurious bit.
//! - **Start a search:** `(highest > 0, not searching, bits == 0)` → `(0, SEARCHING, 0)`. By I2 no
//!   shard has `highest` any more, and by I1 none has more, so `highest` is stale. (With `highest
//!   == 0` and no bits the queue is empty instead, I4.)
//! - **Help the search:** lock a pseudo-random shard whose bit is unset, max its top rank into
//!   `highest` (an empty shard contributes nothing) and set its bit.
//! - **Finish the search** (whoever sees all bits set): `(H, SEARCHING, all bits)` → `(H, all
//!   bits)`, the conservative superset of I2. This is also the result when nothing was found (`H ==
//!   0`): empty shards and shards with rank-`0` items look the same in `highest`, so the bits stay
//!   set and lazy clearing reaches the empty state (I4) once every shard was seen without rank `0`.
//!
//! ## Races considered
//!
//! - *Push during a search* to an already checked shard: the push re-checks it (CAS-max, I3), so
//!   the search can't finish below that push. A push to an unchecked shard is seen by its later
//!   check, or the push itself checks it.
//! - *Search end vs. lower-ranked shards:* the search ends with all bits set although only some
//!   shards have the new `highest`. That's allowed by I2; the spurious bits are cleared lazily by
//!   pops and steals that lock those shards.
//! - *Raise vs. bit clear:* a pop clearing its bit for `highest == h` races with a push raising
//!   `highest` to `h' > h`. The clear is a CAS conditioned on `highest == h`, so it fails and
//!   retries, and then sees `h'` and does nothing.
//! - *Steal vs. search:* a search can only start once all bits are clear. While a thief holds a
//!   victim that has `highest`, the victim's bit can't be cleared (I5) unless `highest` changes. If
//!   `highest` is raised and then drained so that a search starts while the thief still holds both
//!   locks, the thief's publish of its own shard counts as a check of it, and the victim is checked
//!   after the thief released it.
//! - *Contention:* a thread that fails to `try_lock` its shard moves on to another one (and stays
//!   there). After a bounded number of attempts it blocks on a lock instead, so a pop never returns
//!   `None` just because shards were busy (I4).
//!
//! ## Ordering guarantee
//!
//! Without concurrent pushes, a pop returns an item of the highest queued priority (among
//! priorities of equal rank, in any order). With concurrent pushes, a pop returns an item whose
//! priority was the highest at some point during the pop: a concurrently pushed higher priority may
//! be overtaken by items that are already being popped.
//!
//! ## Memory ordering for callers
//!
//! A push that happens before a `pop` call (e.g. via a `SeqCst` fence pairing) is observed by that
//! `pop`: it either returns an item or the queue was empty after the push. `pop` only returns
//! `None` once it read the empty state word, and that word can only be written after every shard
//! that held an item was locked and seen without it.

use std::{
    cell::Cell,
    collections::{BinaryHeap, hash_map::Entry},
    hash::Hash,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

use crossbeam_utils::CachePadded;
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::FxHashMap;

/// A priority that can be summarized in the queue's shared state word.
pub trait QueuePriority {
    /// A rank that is monotone in the priority's order: `a < b` implies
    /// `a.rank() <= b.rank()`. The lowest priority (most items) should have rank `0`, so that
    /// draining it empties the queue without a search for the next priority.
    ///
    /// Ranks don't need to be distinct: priorities of equal rank are only ordered within one shard,
    /// across shards they are treated as equal.
    fn rank(&self) -> u32;
}

/// Unsigned integers up to 32 bits are their own rank.
macro_rules! impl_queue_priority_for_uint {
    ($($ty:ty),*) => {
        $(
            impl QueuePriority for $ty {
                fn rank(&self) -> u32 {
                    u32::from(*self)
                }
            }
        )*
    };
}
impl_queue_priority_for_uint!(u8, u16, u32);

/// The number of shards. Each has one bit in the state word, next to the `SEARCHING` flag and the
/// 32-bit `highest` rank.
pub const SHARD_COUNT: usize = 31;
const ALL_BITS: u64 = (1 << SHARD_COUNT) - 1;
const SEARCHING: u64 = 1 << SHARD_COUNT;
const HIGHEST_SHIFT: u32 = 32;
/// How many shards a thread tries with `try_lock` before it blocks on a lock.
const TRY_LOCK_ATTEMPTS: usize = 4;

fn state_highest(state: u64) -> u32 {
    (state >> HIGHEST_SHIFT) as u32
}

fn state_bits(state: u64) -> u64 {
    state & ALL_BITS
}

fn is_searching(state: u64) -> bool {
    state & SEARCHING != 0
}

fn make_state(highest: u32, searching: bool, bits: u64) -> u64 {
    ((highest as u64) << HIGHEST_SHIFT) | if searching { SEARCHING } else { 0 } | bits
}

fn shard_bit(index: usize) -> u64 {
    1 << index
}

/// Hands out the initial shard of each thread round-robin.
static NEXT_THREAD_SHARD: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The shard this thread pushes to and pops from first. `usize::MAX` until first use.
    static THREAD_SHARD: Cell<usize> = const { Cell::new(usize::MAX) };
    /// State of a per-thread xorshift generator, used to spread steal and search attempts.
    static THREAD_RNG: Cell<u32> = const { Cell::new(0) };
}

fn thread_shard() -> usize {
    THREAD_SHARD.with(|shard| {
        let mut index = shard.get();
        if index == usize::MAX {
            index = NEXT_THREAD_SHARD.fetch_add(1, Ordering::Relaxed) % SHARD_COUNT;
            shard.set(index);
        }
        index
    })
}

fn set_thread_shard(index: usize) {
    THREAD_SHARD.with(|shard| shard.set(index));
}

/// A pseudo-random number in `0..SHARD_COUNT`.
fn random_shard() -> usize {
    THREAD_RNG.with(|rng| {
        let mut x = rng.get();
        if x == 0 {
            // Seed from the thread's shard, so threads start at different points.
            x = (thread_shard() as u32).wrapping_mul(0x9E37_79B9) | 1;
        }
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rng.set(x);
        x as usize % SHARD_COUNT
    })
}

/// One shard: the items queued in it, by priority.
///
/// Invariant: `heap` contains exactly the keys of `buckets`, each once, and no bucket is empty.
struct Shard<P, T> {
    buckets: FxHashMap<P, Vec<T>>,
    heap: BinaryHeap<P>,
}

impl<P: Clone + Ord + Hash + QueuePriority, T> Shard<P, T> {
    fn new() -> Self {
        Self {
            buckets: FxHashMap::default(),
            heap: BinaryHeap::new(),
        }
    }

    fn top_rank(&self) -> Option<u32> {
        self.heap.peek().map(QueuePriority::rank)
    }

    /// Whether the shard is non-empty and its top priority has rank `rank`.
    fn has_rank(&self, rank: u32) -> bool {
        self.top_rank() == Some(rank)
    }

    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    fn push(&mut self, priority: P, item: T) {
        match self.buckets.entry(priority) {
            Entry::Occupied(mut entry) => entry.get_mut().push(item),
            Entry::Vacant(entry) => {
                self.heap.push(entry.key().clone());
                entry.insert(vec![item]);
            }
        }
    }

    /// Appends `items` (not empty) to the bucket of `priority`.
    fn push_many(&mut self, priority: P, mut items: Vec<T>) {
        match self.buckets.entry(priority) {
            Entry::Occupied(mut entry) => entry.get_mut().append(&mut items),
            Entry::Vacant(entry) => {
                self.heap.push(entry.key().clone());
                entry.insert(items);
            }
        }
    }

    /// Removes the newest item of the top priority.
    fn pop_top(&mut self) -> Option<(P, T)> {
        let priority = self.heap.peek()?.clone();
        let bucket = self.buckets.get_mut(&priority).unwrap();
        let item = bucket.pop().unwrap();
        if bucket.is_empty() {
            self.buckets.remove(&priority);
            self.heap.pop();
        }
        Some((priority, item))
    }

    /// Removes half (rounded up) of the items of the top priority, the newest ones.
    fn take_half_top(&mut self) -> Option<(P, Vec<T>)> {
        let priority = self.heap.peek()?.clone();
        let bucket = self.buckets.get_mut(&priority).unwrap();
        let count = bucket.len().div_ceil(2);
        if count == bucket.len() {
            let items = self.buckets.remove(&priority).unwrap();
            self.heap.pop();
            Some((priority, items))
        } else {
            let keep = bucket.len() - count;
            Some((priority, bucket.split_off(keep)))
        }
    }
}

/// What a pop attempt on a locked shard found.
enum ShardPop<P, T> {
    /// An item at the highest priority.
    Item(P, T),
    /// The shard doesn't have the highest priority (any more).
    NotHighest,
    /// The state changed (a search started or the queue became empty); start over.
    Retry,
}

/// A shard behind its lock, on its own cache line.
type PaddedShard<P, T> = CachePadded<Mutex<Shard<P, T>>>;

/// A concurrent priority queue: pops return the highest priority first (see the
/// [crate docs](crate) for the exact ordering guarantee); items of equal priority have no order.
pub struct ConcurrentPriorityQueue<P, T> {
    shards: Box<[PaddedShard<P, T>]>,
    /// `highest` rank, `SEARCHING` flag and shard bits, see the crate docs.
    state: CachePadded<AtomicU64>,
}

impl<P: Clone + Ord + Hash + QueuePriority, T> Default for ConcurrentPriorityQueue<P, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Clone + Ord + Hash + QueuePriority, T> ConcurrentPriorityQueue<P, T> {
    pub fn new() -> Self {
        Self {
            shards: (0..SHARD_COUNT)
                .map(|_| CachePadded::new(Mutex::new(Shard::new())))
                .collect(),
            state: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Whether the queue is empty. Only a snapshot when used concurrently with pushes and pops.
    ///
    /// The state word alone can't always tell: after the last item of a rank above `0` was popped,
    /// it keeps that stale `highest` until the next `pop` searches, and after a search it keeps
    /// spurious bits until they are cleared lazily. So unless the word is `0`, this looks at every
    /// shard (locking each one briefly).
    pub fn is_empty(&self) -> bool {
        self.state.load(Ordering::Acquire) == 0
            || self.shards.iter().all(|shard| shard.lock().is_empty())
    }

    /// Locks the thread's shard, or another one if that is busy. Blocks on a lock only after
    /// [`TRY_LOCK_ATTEMPTS`] failed attempts. A thread that moved to another shard stays there.
    fn lock_some_shard(&self) -> (usize, MutexGuard<'_, Shard<P, T>>) {
        let home = thread_shard();
        let mut index = home;
        for _ in 0..TRY_LOCK_ATTEMPTS {
            if let Some(guard) = self.shards[index].try_lock() {
                if index != home {
                    set_thread_shard(index);
                }
                return (index, guard);
            }
            index = random_shard();
        }
        (home, self.shards[home].lock())
    }

    /// Queues `item` with `priority`.
    pub fn push(&self, priority: P, item: T) {
        let (index, mut shard) = self.lock_some_shard();
        shard.push(priority, item);
        self.publish(index, &shard);
    }

    /// Queues all `items`, like calling [`ConcurrentPriorityQueue::push`] for each of them, but
    /// locks a shard and updates the state word only once for the whole batch.
    ///
    /// The items are collected before the shard is locked, so the caller's iterator never runs
    /// under a shard lock (a panicking iterator can't leave unpublished items behind, and a
    /// reentrant one can't deadlock).
    pub fn push_many(&self, items: impl IntoIterator<Item = (P, T)>) {
        let items: Vec<(P, T)> = items.into_iter().collect();
        if items.is_empty() {
            return;
        }
        let (index, mut shard) = self.lock_some_shard();
        for (priority, item) in items {
            shard.push(priority, item);
        }
        self.publish(index, &shard);
    }

    /// Publishes the top rank of shard `index`, whose lock the caller holds and which is not
    /// empty. Called after inserting into the shard. See "Push" in the crate docs.
    fn publish(&self, index: usize, shard: &Shard<P, T>) {
        let rank = shard
            .top_rank()
            .expect("a shard is published after inserting into it");
        let bit = shard_bit(index);
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            let highest = state_highest(state);
            let new_state = if is_searching(state) {
                make_state(highest.max(rank), true, state_bits(state) | bit)
            } else if rank > highest {
                make_state(rank, false, bit)
            } else if rank == highest && state & bit == 0 {
                state | bit
            } else {
                return;
            };
            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if is_searching(new_state) && state_bits(new_state) == ALL_BITS {
                        self.finish_search();
                    }
                    return;
                }
                Err(actual) => state = actual,
            }
        }
    }

    /// Clears the bit of shard `index`, whose lock the caller holds and which doesn't have rank
    /// `highest`, if `highest` is still the state's highest rank and no search is in progress.
    fn clear_bit_if_highest(&self, index: usize, highest: u32) {
        let bit = shard_bit(index);
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            if is_searching(state) || state_highest(state) != highest || state & bit == 0 {
                return;
            }
            match self.state.compare_exchange_weak(
                state,
                state & !bit,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(actual) => state = actual,
            }
        }
    }

    /// Ends a search once every shard was checked.
    fn finish_search(&self) {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            if !is_searching(state) || state_bits(state) != ALL_BITS {
                return;
            }
            let new_state = make_state(state_highest(state), false, ALL_BITS);
            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(actual) => state = actual,
            }
        }
    }

    /// Checks one unchecked shard for the running search, see "Help the search" in the crate docs.
    fn help_search(&self, state: u64) {
        let unchecked = !state & ALL_BITS;
        if unchecked == 0 {
            self.finish_search();
            return;
        }
        let start = random_shard();
        let mut first_unchecked = None;
        for offset in 0..SHARD_COUNT {
            let index = (start + offset) % SHARD_COUNT;
            if unchecked & shard_bit(index) == 0 {
                continue;
            }
            first_unchecked.get_or_insert(index);
            if let Some(shard) = self.shards[index].try_lock() {
                self.check_shard(index, &shard);
                return;
            }
        }
        // All unchecked shards are busy: wait for one, so the search makes progress.
        let index = first_unchecked.unwrap();
        let shard = self.shards[index].lock();
        self.check_shard(index, &shard);
    }

    /// Records the check of shard `index`, whose lock the caller holds, for the running search.
    fn check_shard(&self, index: usize, shard: &Shard<P, T>) {
        // An empty shard contributes nothing to the maximum.
        let rank = shard.top_rank().unwrap_or(0);
        let bit = shard_bit(index);
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            if !is_searching(state) || state & bit != 0 {
                return;
            }
            let new_state = make_state(
                state_highest(state).max(rank),
                true,
                state_bits(state) | bit,
            );
            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if state_bits(new_state) == ALL_BITS {
                        self.finish_search();
                    }
                    return;
                }
                Err(actual) => state = actual,
            }
        }
    }

    /// Pops an item of the highest priority. Only returns `None` when the queue is empty (I4).
    pub fn pop(&self) -> Option<(P, T)> {
        loop {
            let state = self.state.load(Ordering::Acquire);
            if state == 0 {
                return None;
            }
            if is_searching(state) {
                self.help_search(state);
                continue;
            }
            if state_bits(state) == 0 {
                // No shard has `highest` (I2), and `highest > 0` (else `state == 0`): search for
                // the new highest priority.
                let _ = self.state.compare_exchange(
                    state,
                    SEARCHING,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                );
                continue;
            }
            let (index, mut shard) = self.lock_some_shard();
            match self.pop_locked(index, &mut shard) {
                ShardPop::Item(priority, item) => return Some((priority, item)),
                ShardPop::Retry => continue,
                ShardPop::NotHighest => {}
            }
            if let Some(item) = self.steal(index, &mut shard) {
                return Some(item);
            }
        }
    }

    /// Pops an item at the highest priority from shard `index`, whose lock the caller holds.
    fn pop_locked(&self, index: usize, shard: &mut Shard<P, T>) -> ShardPop<P, T> {
        // Re-read under the lock: never pop below the current highest priority.
        let state = self.state.load(Ordering::Acquire);
        if state == 0 || is_searching(state) {
            return ShardPop::Retry;
        }
        let highest = state_highest(state);
        if !shard.has_rank(highest) {
            self.clear_bit_if_highest(index, highest);
            return ShardPop::NotHighest;
        }
        let (priority, item) = shard.pop_top().unwrap();
        if !shard.has_rank(highest) {
            self.clear_bit_if_highest(index, highest);
        }
        ShardPop::Item(priority, item)
    }

    /// Steals half of the items at the highest priority from a shard whose bit is set into shard
    /// `index` (whose lock the caller holds), and pops one of them. See "Steal" in the crate docs.
    fn steal(&self, index: usize, shard: &mut Shard<P, T>) -> Option<(P, T)> {
        let state = self.state.load(Ordering::Acquire);
        if state == 0 || is_searching(state) {
            return None;
        }
        let highest = state_highest(state);
        let candidates = state_bits(state) & !shard_bit(index);
        let start = random_shard();
        for offset in 0..SHARD_COUNT {
            let victim_index = (start + offset) % SHARD_COUNT;
            if candidates & shard_bit(victim_index) == 0 {
                continue;
            }
            let Some(mut victim) = self.shards[victim_index].try_lock() else {
                continue;
            };
            let state = self.state.load(Ordering::Acquire);
            if is_searching(state) || state_highest(state) != highest {
                return None;
            }
            if !victim.has_rank(highest) {
                self.clear_bit_if_highest(victim_index, highest);
                continue;
            }
            let (priority, items) = victim.take_half_top().unwrap();
            shard.push_many(priority, items);
            // Publish ours before clearing the victim's bit, so the bits don't run empty in
            // between.
            self.publish(index, shard);
            if !victim.has_rank(highest) {
                self.clear_bit_if_highest(victim_index, highest);
            }
            drop(victim);
            return match self.pop_locked(index, shard) {
                ShardPop::Item(priority, item) => Some((priority, item)),
                // The state changed: let `pop` start over.
                ShardPop::NotHighest | ShardPop::Retry => None,
            };
        }
        None
    }
}

#[cfg(test)]
mod tests;

use std::{
    collections::HashSet,
    sync::{LazyLock, Mutex},
    time::Duration,
};

use egg::{Id, Subst, Symbol};

pub static PROCESSED: LazyLock<Mutex<HashSet<(Symbol, Id, Subst)>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

// External Timeout
pub const TIME_LIMIT: Duration = Duration::from_millis(10000);
// Number of iterations after which the extractor runs
pub const N: usize = 1;
// Max PiiGraph size for Slow Extractor Algorithm
pub const SLOW_LIMIT_START: usize = 10;
pub const SLOW_LIMIT_CUTOFF: usize = 15;
// Ratio of termination time to time since last improvement
pub const SATURATION_FACTOR: u32 = 2;

// Debug field to slow down the pipeline to test termination conditions
pub const SLEEP_TIME: Duration = Duration::from_millis(0);

pub mod cost;
pub mod ir;
pub mod isel;
pub mod malloc;

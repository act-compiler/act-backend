use std::time::Duration;

// The whole search's time
pub const TIME_LIMIT: Duration = Duration::from_millis(1000);
// The e-graph's nodes, past which no rewrite runs
pub const NODE_LIMIT: usize = 10000;
// The rewrites' iterations
pub const ITERATION_LIMIT: usize = 30;

// Max PiiGraph size for Slow Extractor Algorithm
pub const SLOW_LIMIT_START: usize = 10;
pub const SLOW_LIMIT_CUTOFF: usize = 15;
// Ratio of termination time to time since last improvement
pub const SATURATION_FACTOR: u32 = 10;

pub mod cost;
pub mod ir;
pub mod isel;
pub mod malloc;

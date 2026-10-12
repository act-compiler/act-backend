use egg::Rewrite;

use crate::ir::egraph::{TensorInfo, TensorOp};
use crate::isel::rewrites::applier::get_applier;

pub fn get_rewrites() -> Vec<Rewrite<TensorOp, TensorInfo>> {
  // The rules are read into the binary when it is built
  let ir2ir = include_str!("ir2ir_rewrites.txt");
  let ir2ir: Vec<&str> = ir2ir.split('\n').filter(|s| !s.is_empty()).collect();

  let ir2isa = include_str!("ir2isa_rewrites.txt");
  let ir2isa: Vec<&str> = ir2isa.split('\n').filter(|s| !s.is_empty()).collect();

  let mut rewrites: Vec<Rewrite<TensorOp, TensorInfo>> = vec![];
  for rewrite_str in [ir2ir, ir2isa].concat().iter() {
    rewrites.push(get_applier(rewrite_str));
  }
  rewrites
}

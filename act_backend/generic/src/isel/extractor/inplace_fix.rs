use std::collections::{HashMap, HashSet};

use crate::ir::pii::PiiGraph;

/// Give an instruction overwriting a shared operand its own copy
/// Err where that operand is pinned in HBM
pub fn fix_inplace(pii: &mut PiiGraph) -> Result<(), String> {
  let mut visited: HashSet<usize> = HashSet::new();
  let mut copied = false;
  copy_overwritten(pii, pii.root, &mut visited, &mut copied)?;
  // A copy goes last, out of topological order
  if copied {
    compact(pii);
  }
  Ok(())
}

/// A node's uses, so f(x, x) uses x twice
fn count_uses(pii: &PiiGraph, target: usize) -> usize {
  pii
    .nodes
    .iter()
    .flat_map(|node| &node.children)
    .filter(|&&child| child == target)
    .count()
}

/// Copy each shared operand overwritten at or below a node
fn copy_overwritten(
  pii: &mut PiiGraph,
  node: usize,
  visited: &mut HashSet<usize>,
  copied: &mut bool,
) -> Result<(), String> {
  if !visited.insert(node) {
    return Ok(());
  }

  let inplace = pii.nodes[node].op.inplace_children();
  let children = pii.nodes[node].children.clone();
  for (i, &child) in children.iter().enumerate() {
    // All false for a plain instruction, empty for an operator
    let overwritten = inplace.get(i) == Some(&true);
    if overwritten && count_uses(pii, child) > 1 {
      // A copy is a node, never storage pinned in HBM
      if pii.nodes[child].hbm_offset.is_some() {
        return Err("an overwritten operand has another reader and is pinned in HBM".to_string());
      }
      let copy = copy_node(pii, child);
      pii.nodes[node].children[i] = copy;
      *copied = true;
      copy_overwritten(pii, copy, visited, copied)?;
    } else {
      copy_overwritten(pii, child, visited, copied)?;
    }
  }
  Ok(())
}

/// A copy of a node, added last
fn copy_node(pii: &mut PiiGraph, node: usize) -> usize {
  let id = pii.nodes.len();
  let mut copy = pii.nodes[node].clone();
  copy.id = id;
  pii.nodes.push(copy);
  id
}

/// Renumber in topological order from the root, dropping dead nodes
fn compact(pii: &mut PiiGraph) {
  let mut order: Vec<usize> = vec![];
  postorder(pii, pii.root, &mut HashSet::new(), &mut order);

  // Old node index -> new
  let remap: HashMap<usize, usize> = order
    .iter()
    .enumerate()
    .map(|(new_id, &old_id)| (old_id, new_id))
    .collect();
  let mut nodes: Vec<_> = order
    .iter()
    .map(|&old_id| pii.nodes[old_id].clone())
    .collect();
  for (new_id, node) in nodes.iter_mut().enumerate() {
    node.id = new_id;
    node.children = node.children.iter().map(|child| remap[child]).collect();
  }

  pii.root = nodes.len() - 1;
  pii.nodes = nodes;
}

/// Each node after its children, from a node down
fn postorder(pii: &PiiGraph, node: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {
  if !visited.insert(node) {
    return;
  }
  for &child in pii.nodes[node].children.iter() {
    postorder(pii, child, visited, order);
  }
  order.push(node);
}

use egg::*;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ir::buffer::Buffer;
use crate::ir::egraph::{TensorInfo, TensorOp};
use crate::ir::pii::PiiGraph;

use crate::ir::buffer::buffer_assignment as get_bufs;
use crate::isel::extractor::utils::{annotate_constants, detect_constants, get_hbm_offset};

/// Fast extraction: one pii graph, each e-class once per buffer
pub fn extract_fast(
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  root: Id,
  inputs: &HashSet<Id>,
  hbm_offsets: &[(Option<Id>, i32)],
) -> Vec<PiiGraph> {
  // Bottom-up traversal to detect compile-time constants
  let constants = detect_constants(egraph, inputs);
  // Annotate the e-graph with DetectedConst nodes and mark analysis data.
  annotate_constants(egraph, &constants);

  // Top-down extraction; e-node order keeps the numbering stable
  let mut selected: BTreeMap<(TensorOp, Buffer), (Id, TensorInfo)> = BTreeMap::new();
  let ret = get_pii_graph(
    egraph,
    root,
    Buffer::HBM,
    inputs,
    &mut HashSet::new(),
    &mut selected,
  );
  if ret {
    vec![select2pii_graph(&selected, hbm_offsets)]
  } else {
    vec![]
  }
}

/// Pre-order extraction: whether one is rooted at eclass
fn get_pii_graph(
  egraph: &EGraph<TensorOp, TensorInfo>,
  eclass: Id,
  target_buf: Buffer,
  inputs: &HashSet<Id>,
  path: &mut HashSet<(Id, Buffer)>,
  selected: &mut BTreeMap<(TensorOp, Buffer), (Id, TensorInfo)>,
) -> bool {
  // An e-class in another buffer is another node
  if path.contains(&(eclass, target_buf)) {
    // no cycles allowed
    return false;
  }

  for en in &egraph[eclass].nodes {
    // Memoization case: a selected e-node, at a cost to completeness
    if selected.contains_key(&(en.clone(), target_buf)) {
      return true;
    }
  }

  'en_loop: for en in &egraph[eclass].nodes {
    // A snapshot to roll back to if this e-node fails
    let snapshot = selected.clone();
    // Base case: input e-node
    if inputs.contains(&eclass) && target_buf == Buffer::HBM {
      selected.insert(
        (en.clone(), target_buf),
        (eclass, egraph[eclass].data.clone()),
      );
      return true;
    }
    // Base case: detected constant e-node
    if egraph[eclass].data.is_const && en.is_detected_const() && target_buf == Buffer::HBM {
      selected.insert(
        (en.clone(), target_buf),
        (eclass, egraph[eclass].data.clone()),
      );
      return true;
    }

    // Only consider instruction e-nodes
    if let Some(bufs) = get_bufs(en) {
      if bufs[0] != Buffer::ANY && bufs[0] != target_buf {
        continue;
      }

      path.insert((eclass, target_buf));
      for (i, child_ec) in en.children().iter().enumerate() {
        let out_buf = child_buf(&bufs, target_buf, i);
        if !get_pii_graph(egraph, *child_ec, out_buf, inputs, path, selected) {
          // Roll back any selections that happened while attempting this e-node
          selected.clear();
          for (k, v) in snapshot.iter() {
            selected.insert(k.clone(), v.clone());
          }
          // Remove current e-class from path before trying next e-node
          path.remove(&(eclass, target_buf));
          continue 'en_loop;
        }
      }
      path.remove(&(eclass, target_buf));

      selected.insert(
        (en.clone(), target_buf),
        (eclass, egraph[eclass].data.clone()),
      );
      return true;
    }
  }
  false
}

/// The selected e-nodes as a pii graph, built bottom-up
fn select2pii_graph(
  selected: &BTreeMap<(TensorOp, Buffer), (Id, TensorInfo)>,
  hbm_offsets: &[(Option<Id>, i32)],
) -> PiiGraph {
  let mut pii_graph = PiiGraph::default();

  let mut to_remove: Vec<(TensorOp, Buffer)> = vec![];
  // Map e-class and buffer -> node index in pii graph
  let mut eclass_map: HashMap<(Id, Buffer), usize> = HashMap::new();

  // Add leaf e-nodes to the pii graph
  for (key, (ec_id, info)) in selected.iter() {
    let (en, buf) = key;
    if en.is_leaf() {
      let new_id = pii_graph.add_node(
        en.clone(),
        info.clone(),
        vec![],
        get_hbm_offset(hbm_offsets, *ec_id),
      );
      eclass_map.insert((*ec_id, *buf), new_id);
      to_remove.push(key.clone());
    }
  }

  // Add first candidate e-node to the pii graph
  while to_remove.len() < selected.len() {
    let mut progress = false;
    for (key, (ec_id, info)) in selected.iter() {
      if to_remove.contains(key) {
        continue;
      }
      let (en, buf) = key;
      let bufs = get_bufs(en).unwrap();
      let children: Option<Vec<usize>> = en
        .children()
        .iter()
        .enumerate()
        .map(|(i, child_ec)| {
          eclass_map
            .get(&(*child_ec, child_buf(&bufs, *buf, i)))
            .copied()
        })
        .collect();
      if let Some(children) = children {
        let new_id = pii_graph.add_node(
          en.clone(),
          info.clone(),
          children,
          get_hbm_offset(hbm_offsets, *ec_id),
        );
        eclass_map.insert((*ec_id, *buf), new_id);
        to_remove.push(key.clone());
        progress = true;
        break;
      }
    }
    if !progress {
      // No node left has all its children added
      debug_assert!(false, "select2pii_graph: possible infinite loop");
      break;
    }
  }

  pii_graph
}

/// Where an e-node selected in `buf` needs its `i`th child
fn child_buf(bufs: &[Buffer], buf: Buffer, i: usize) -> Buffer {
  if bufs[0] != Buffer::ANY {
    bufs[1 + i]
  } else {
    // A buffer-agnostic e-node takes children from its own buffer
    buf
  }
}

use std::fs::File;
use std::io::Write;

use crate::ir::buffer::Buffer;
use crate::ir::egraph::{TensorInfo, TensorOp};

use crate::ir::buffer::buffer_assignment as get_bufs;

#[derive(Debug, Clone)]
pub struct PiiNode {
    pub id: usize,
    pub op: TensorOp,
    pub info: TensorInfo,
    pub buffer: Buffer,
    pub hbm_offset: Option<i32>,
    pub children: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct PiiGraph {
    pub nodes: Vec<PiiNode>,
}

impl PiiGraph {
    pub fn add_node(
        &mut self,
        op: TensorOp,
        info: TensorInfo,
        children: Vec<usize>,
        hbm_offset: Option<i32>,
    ) -> usize {
        let mut buffer = match get_bufs(&op) {
            Some(bufs) => bufs[0],
            None => panic!("Not a valid pii node operation: {:?}", op),
        };
        if buffer == Buffer::ANY {
            // If all children have the same buffer, we use that buffer else panic
            let child_buffers: Vec<Buffer> =
                children.iter().map(|&c| self.nodes[c].buffer).collect();
            if child_buffers.is_empty() {
                panic!("ANY buffer requires at least one child to infer buffer type");
            }
            let first_buf = child_buffers[0];
            if child_buffers.iter().all(|&b| b == first_buf) {
                buffer = first_buf;
            } else {
                panic!("Cannot infer buffer: children have different buffers");
            }
        }

        let idx = self.nodes.len();
        self.nodes.push(PiiNode {
            id: idx,
            op,
            info,
            children,
            buffer,
            hbm_offset,
        });
        idx
    }

    fn shape_to_string(shape: &Vec<i32>) -> String {
        let s: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("[{}]", s.join(","))
    }

    pub fn save(&self, path: &std::path::PathBuf) {
        let mut file = File::create(path).expect("Unable to create file");
        write!(file, "{}", self).expect("Unable to write data");
    }
}

impl std::fmt::Display for PiiGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for n in self.nodes.iter() {
            let dtype = n.info.dtype.to_string();
            let shape = Self::shape_to_string(&n.info.shape);
            let op_str = n.op.to_string();
            let children_str: Vec<String> = n.children.iter().map(|c| format!("t{}", c)).collect();
            writeln!(
                f,
                "t{}: {}[{}] = {}{} {}({})",
                n.id,
                n.buffer,
                n.hbm_offset.unwrap_or(-1),
                dtype,
                shape,
                op_str,
                children_str.join(", ")
            )?;
        }
        Ok(())
    }
}

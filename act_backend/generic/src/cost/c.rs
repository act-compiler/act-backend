use std::{ffi::CString, path::Path};

use libc::{c_char, c_double, c_int};

// The cost model linked into the backend
unsafe extern "C" {
  fn act_cost(asm_path: *const c_char, cost: *mut c_double) -> c_int;
}

/// A candidate's cost, or why the cost model gave none
pub fn c_bridge(asm_path: &Path) -> Result<f64, String> {
  let path = CString::new(asm_path.to_str().unwrap()).unwrap();
  let mut cost: c_double = 0.0;

  let status = unsafe { act_cost(path.as_ptr(), &mut cost) };
  if status != 0 {
    return Err(format!("returned {}", status));
  }
  // A cost that orders no candidate is no cost
  if !cost.is_finite() {
    return Err(format!("gave the non-finite cost {}", cost));
  }
  Ok(cost)
}

use std::{
  ffi::{CStr, CString},
  path::Path,
};

use libc::{c_char, size_t};

// The most bytes of a reason the allocator hands back
const REASON_SIZE: usize = 1024;

#[link(name = "malloc_static")]
unsafe extern "C" {
  fn act_malloc(
    pii_path: *const c_char,
    metadata_path: *const c_char,
    asm_path: *const c_char,
    reason: *mut c_char,
    reason_size: size_t,
  ) -> i32;
}

/// Write one candidate's assembly, or say why it has none
pub fn cpp_bridge(pii_path: &Path, metadata_path: &Path, asm_path: &Path) -> Result<(), String> {
  let pii_path = CString::new(pii_path.to_str().unwrap()).unwrap();
  let metadata_path = CString::new(metadata_path.to_str().unwrap()).unwrap();
  let asm_path = CString::new(asm_path.to_str().unwrap()).unwrap();
  let mut reason = vec![0u8; REASON_SIZE];

  let status = unsafe {
    act_malloc(
      pii_path.as_ptr(),
      metadata_path.as_ptr(),
      asm_path.as_ptr(),
      reason.as_mut_ptr() as *mut c_char,
      reason.len(),
    )
  };
  if status == 0 {
    return Ok(());
  }
  // The allocator ends its reason with a zero byte
  let reason = CStr::from_bytes_until_nul(&reason).unwrap();
  Err(reason.to_string_lossy().into_owned())
}

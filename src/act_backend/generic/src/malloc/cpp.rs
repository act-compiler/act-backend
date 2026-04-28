use std::{ffi::CString, path::PathBuf};

use libc::c_char;

#[link(name = "malloc_static")]
unsafe extern "C" {
    fn act_malloc(
        pii_path: *const c_char,
        metadata_path: *const c_char,
        asm_path: *const c_char,
    ) -> i32;
}

pub fn cpp_bridge(pii_path: &PathBuf, metadata_path: &PathBuf, asm_path: &PathBuf) -> bool {
    let pii_path = CString::new(pii_path.to_str().unwrap()).unwrap();
    let metadata_path = CString::new(metadata_path.to_str().unwrap()).unwrap();
    let asm_path = CString::new(asm_path.to_str().unwrap()).unwrap();

    unsafe {
        match act_malloc(pii_path.as_ptr(), metadata_path.as_ptr(), asm_path.as_ptr()) {
            0 => true,
            _ => false,
        }
    }
}

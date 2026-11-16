// large, zero-initialized allocations with transparent huge page support

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

// zeroed buffer, uses huge pages when possible
pub struct HugeBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> HugeBuffer<T> {
    pub fn zeroed(len: usize) -> Self {
        assert!(len > 0, "zero-sized huge allocation");
        let bytes = std::mem::size_of::<T>().checked_mul(len).expect("huge allocation size overflow");
        let ptr = raw_alloc(bytes);
        Self {
            ptr: NonNull::new(ptr as *mut T).expect("out of memory"),
            len,
            _marker: PhantomData,
        }
    }

    // leak buffer for static lifetime
    pub fn leak(self) -> *mut T {
        let ptr = self.ptr.as_ptr();
        std::mem::forget(self);
        ptr
    }

    #[inline(always)]
    pub fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr.as_ptr() }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Deref for HugeBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for HugeBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for HugeBuffer<T> {
    fn drop(&mut self) {
        raw_free(self.ptr.as_ptr() as *mut u8, std::mem::size_of::<T>() * self.len);
    }
}

// single zeroed object in huge pages
pub struct HugeBox<T> {
    buffer: HugeBuffer<T>,
}

impl<T> HugeBox<T> {
    pub fn zeroed() -> Self {
        Self { buffer: HugeBuffer::zeroed(1) }
    }

    // leaks the block and returns a raw pointer
    pub fn leak(self) -> *mut T {
        self.buffer.leak()
    }

    #[inline(always)]
    pub fn as_mut(&mut self) -> &mut T {
        self.buffer.as_mut()
    }
}

impl<T> Deref for HugeBox<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        &self.buffer[0]
    }
}

impl<T> DerefMut for HugeBox<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.buffer[0]
    }
}

unsafe impl<T: Send> Send for HugeBuffer<T> {}
unsafe impl<T: Sync> Sync for HugeBuffer<T> {}

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;
    use std::sync::Once;

    type Handle = *mut c_void;
    type Bool = i32;
    type Dword = u32;

    #[repr(C)]
    struct Luid {
        low: Dword,
        high: i32,
    }

    #[repr(C)]
    struct TokenPrivileges {
        count: Dword,
        luid: Luid,
        attributes: Dword,
    }

    const TOKEN_ADJUST_PRIVILEGES: Dword = 0x0020;
    const TOKEN_QUERY: Dword = 0x0008;
    const SE_PRIVILEGE_ENABLED: Dword = 0x0002;
    const MEM_COMMIT: Dword = 0x1000;
    const MEM_RESERVE: Dword = 0x2000;
    const MEM_RELEASE: Dword = 0x8000;
    const MEM_LARGE_PAGES: Dword = 0x2000_0000;
    const PAGE_READWRITE: Dword = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> Handle;
        fn CloseHandle(handle: Handle) -> Bool;
        fn VirtualAlloc(address: *mut c_void, size: usize, kind: Dword, protect: Dword) -> *mut c_void;
        fn VirtualFree(address: *mut c_void, size: usize, kind: Dword) -> Bool;
        fn GetLargePageMinimum() -> usize;
    }

    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn OpenProcessToken(process: Handle, access: Dword, token: *mut Handle) -> Bool;
        fn LookupPrivilegeValueW(system: *const u16, name: *const u16, luid: *mut Luid) -> Bool;
        fn AdjustTokenPrivileges(
            token: Handle,
            disable_all: Bool,
            new_state: *const TokenPrivileges,
            buffer_length: Dword,
            previous: *mut c_void,
            return_length: *mut Dword,
        ) -> Bool;
    }

    static PRIVILEGE: Once = Once::new();

    fn enable_large_pages() {
        PRIVILEGE.call_once(|| unsafe {
            let mut token: Handle = std::ptr::null_mut();
            if OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &mut token) == 0 {
                return;
            }

            let name: Vec<u16> = "SeLockMemoryPrivilege".encode_utf16().chain(std::iter::once(0)).collect();
            let mut luid = Luid { low: 0, high: 0 };
            if LookupPrivilegeValueW(std::ptr::null(), name.as_ptr(), &mut luid) != 0 {
                let state = TokenPrivileges {
                    count: 1,
                    luid,
                    attributes: SE_PRIVILEGE_ENABLED,
                };
                AdjustTokenPrivileges(token, 0, &state, 0, std::ptr::null_mut(), std::ptr::null_mut());
            }
            CloseHandle(token);
        });
    }

    pub fn alloc(bytes: usize) -> *mut u8 {
        enable_large_pages();
        unsafe {
            let page = GetLargePageMinimum();
            if page > 0 && bytes >= page / 2 {
                let rounded = bytes.div_ceil(page) * page;
                let ptr = VirtualAlloc(std::ptr::null_mut(), rounded, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
                if !ptr.is_null() {
                    return ptr as *mut u8;
                }
            }

            let ptr = VirtualAlloc(std::ptr::null_mut(), bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            assert!(!ptr.is_null(), "VirtualAlloc failed");
            ptr as *mut u8
        }
    }

    pub fn free(ptr: *mut u8) {
        unsafe {
            VirtualFree(ptr as *mut c_void, 0, MEM_RELEASE);
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::ffi::c_void;

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 2;
    const MAP_ANONYMOUS: i32 = 0x20;
    const MADV_HUGEPAGE: i32 = 14;
    const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;

    unsafe extern "C" {
        fn mmap(address: *mut c_void, length: usize, protection: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
        fn madvise(address: *mut c_void, length: usize, advice: i32) -> i32;
        fn munmap(address: *mut c_void, length: usize) -> i32;
    }

    pub fn alloc(bytes: usize) -> *mut u8 {
        unsafe {
            let ptr = mmap(std::ptr::null_mut(), bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            assert!(ptr != MAP_FAILED, "mmap failed");
            // best effort: kernels with THP disabled simply ignore this
            madvise(ptr, bytes, MADV_HUGEPAGE);
            ptr as *mut u8
        }
    }

    pub fn free(ptr: *mut u8, bytes: usize) {
        unsafe {
            munmap(ptr as *mut c_void, bytes);
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod platform {
    pub fn alloc(bytes: usize) -> *mut u8 {
        let layout = std::alloc::Layout::from_size_align(bytes, 64).expect("invalid allocation");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory");
        ptr
    }

    pub fn free(ptr: *mut u8, bytes: usize) {
        let layout = std::alloc::Layout::from_size_align(bytes, 64).expect("invalid allocation");
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn raw_alloc(bytes: usize) -> *mut u8 {
    platform::alloc(bytes)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn raw_free(ptr: *mut u8, bytes: usize) {
    platform::free(ptr, bytes);
}

#[cfg(windows)]
fn raw_alloc(bytes: usize) -> *mut u8 {
    platform::alloc(bytes)
}

#[cfg(windows)]
fn raw_free(ptr: *mut u8, _bytes: usize) {
    platform::free(ptr)
}

#[cfg(target_os = "linux")]
fn raw_alloc(bytes: usize) -> *mut u8 {
    platform::alloc(bytes)
}

#[cfg(target_os = "linux")]
fn raw_free(ptr: *mut u8, bytes: usize) {
    platform::free(ptr, bytes)
}

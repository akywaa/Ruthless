use std::sync::atomic::{AtomicBool, Ordering};

pub static BIND_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_bind_enabled(enabled: bool) {
    BIND_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn bind_thread(thread_id: usize) {
    if !BIND_ENABLED.load(Ordering::Relaxed) {
        return;
    }

    if let Some(core_ids) = core_affinity::get_core_ids() {
        if !core_ids.is_empty() {
            // Pin the thread to a specific logical core. This prevents the OS scheduler
            // from migrating the thread across NUMA nodes, heavily increasing cache hits.
            let core = core_ids[thread_id % core_ids.len()];
            core_affinity::set_for_current(core);
        }
    }
}

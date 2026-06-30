/// These macros are a workaround for rayon failing on wasm targets - they replace the
/// par_ version with the standard single-threaded version.

#[macro_export]
macro_rules! par_iter {
    ($collection:expr) => {{
        #[cfg(target_family = "wasm")]
        {
            $collection.par_iter()
        }

        #[cfg(not(target_family = "wasm"))]
        {
            $collection.par_iter()
        }
    }};
}

#[macro_export]
macro_rules! par_sort_by {
    ($collection:expr, $cmp:expr) => {{
        #[cfg(target_family = "wasm")]
        {
            $collection.par_sort_by($cmp)
        }

        #[cfg(not(target_family = "wasm"))]
        {
            $collection.par_sort_by($cmp)
        }
    }};
}

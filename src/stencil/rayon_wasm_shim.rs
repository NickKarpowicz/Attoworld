#[macro_export]
macro_rules! par_iter {
    ($collection:expr) => {{
        #[cfg(target_family = "wasm")]
        {
            $collection.iter()
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
            $collection.sort_by($cmp)
        }

        #[cfg(not(target_family = "wasm"))]
        {
            $collection.par_sort_by($cmp)
        }
    }};
}

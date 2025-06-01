use pyo3::prelude::*;

/// Function written in Rust for better performance and correctness.
#[pymodule]
#[pyo3(name = "attoworld_rs")]
fn attoworld_rs_test(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_hello, m)?)?;
    Ok(())
}

/// Test function to make sure the Rust module is working
#[pyfunction]
fn rust_hello() -> PyResult<()> {
    println!("Hi from Rust!");
    Ok(())
}

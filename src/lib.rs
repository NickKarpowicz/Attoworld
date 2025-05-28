use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "attoworld_rs")]
fn attoworld_rs_test(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_hello, m)?)?;
    Ok(())
}

#[pyfunction]
fn rust_hello() -> PyResult<()> {
    println!("Hi from Rust!");
    Ok(())
}

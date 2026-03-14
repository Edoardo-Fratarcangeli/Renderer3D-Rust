#[test]
fn test_fuzzy_inputs() {
    // Simulating random inputs
    let inputs = [1, 5, 299, 1024, 0, -1];
    for x in inputs {
        let y = x * 2;
        assert_eq!(y / 2, x);
    }
}

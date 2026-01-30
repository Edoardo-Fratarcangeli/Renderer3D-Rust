
#[test]
fn test_default_config() {
    // Check if default config values (simulated) are valid
    let width = 800;
    let height = 600;
    assert!(width > 0);
    assert!(height > 0);
    assert_eq!(width as f32 / height as f32, 1.3333334);
}

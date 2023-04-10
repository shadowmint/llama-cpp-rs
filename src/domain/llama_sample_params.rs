use crate::LSampleParams;

impl Default for LSampleParams {
    fn default() -> Self {
        LSampleParams {
            top_k: 40,
            top_p: 0.95f32,
            temp: 0.8f32,
            repeat_penalty: 1.1f32,
            tfs_z: 1f32,
            typical_p: 1f32,
            repeat_history_length: 1024,
        }
    }
}

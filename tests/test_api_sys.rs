use llama_cpp_sys::*;
use std::ffi::{CStr, CString};
use std::io::Write;
use std::ptr;
use std::ptr::null_mut;

#[test]
pub fn main() {
    unsafe {
        let mut params = llama_context_default_params();
        params.seed = 0;
        params.n_ctx = 512;
        params.n_parts = -1;
        params.f16_kv = true;
        params.use_mlock = false;
        params.vocab_only = false;
        params.logits_all = false;
        params.embedding = false;
        params.progress_callback = None;

        let model_path = CString::new("models/13B/model.bin").unwrap();
        let ctx = llama_init_from_file(model_path.as_ptr(), params);
        assert_ne!(ctx, null_mut());
        println!("loaded model");

        let prompt = "bob is a space pilot. Alice is a potato. This is a conversation between bob and alice:";
        let c_prompt = CString::new(prompt).unwrap();
        let c_prompt_bytes = c_prompt.as_bytes_with_nul();
        let c_prompt_ptr = &c_prompt_bytes[0] as *const u8 as *const i8;

        let mut embd_inp: Vec<llama_token> = vec![0; prompt.len() + 1];
        let embd_inp_ptr: *mut i32 = &mut embd_inp[0] as *mut i32;
        let embd_inp_len = embd_inp.len() as i32;
        let prompt_token_count =
            llama_tokenize(ctx, c_prompt_ptr, embd_inp_ptr, embd_inp_len, true);
        assert!(prompt_token_count > 0);
        embd_inp.truncate(prompt_token_count as usize);
        println!("prompt length: {}", prompt_token_count);

        let n_ctx = llama_n_ctx(ctx);
        println!("model context length: {}", n_ctx);

        let mut token_stream = embd_inp.clone();
        let num_predict = 128;
        assert!(num_predict + token_stream.len() < n_ctx as usize);

        println!("prompt:");
        for token in token_stream.iter() {
            let c_buf = llama_token_to_str(ctx, *token);
            let c_str: &CStr = CStr::from_ptr(c_buf);
            let str_slice: &str = c_str.to_str().unwrap();
            print!("{}", str_slice);
        }
        println!("\n\ngenerating...\n");

        let sample_worker_threads = 8;
        let sample_size = 64;
        let mut gen_buffer = vec![0; sample_size];

        // Initialize with prompt
        let eval_result = llama_eval(
            ctx,
            token_stream.as_ptr(),
            token_stream.len() as i32,
            0,
            sample_worker_threads,
        );
        assert_eq!(eval_result, 0);

        for _ in 0..num_predict {
            let top_k = 40;
            let top_p = 0.95f32;
            let temp = 0.8f32;
            let repeat_penalty = 1.1f32;

            // Build a query buffer
            let gen_buffer_ptr = gen_buffer.as_mut_ptr();
            ptr::write_bytes(gen_buffer_ptr, 0, gen_buffer.len());
            if token_stream.len() < sample_size {
                let end_partial = &mut gen_buffer[sample_size - token_stream.len()..];
                end_partial.copy_from_slice(&token_stream);
            } else {
                gen_buffer.copy_from_slice(&token_stream[(token_stream.len() - sample_size)..])
            }

            // Invoke model
            let last_real_token = &gen_buffer[gen_buffer.len() - 1] as *const i32;
            let num_tokens_to_generate = 1;
            let past_tokens = token_stream.len() as i32;
            let eval_result = llama_eval(
                ctx,
                last_real_token,
                num_tokens_to_generate,
                past_tokens,
                sample_worker_threads,
            );
            assert_eq!(eval_result, 0);

            // Sample result
            let id = llama_sample_top_p_top_k(
                ctx,
                gen_buffer.as_mut_ptr(),
                sample_size as i32,
                top_k,
                top_p,
                temp,
                repeat_penalty,
            );

            if id == llama_token_eos() {
                println!("Received end of stream");
                break;
            }

            let c_buf = llama_token_to_str(ctx, id);
            let c_str: &CStr = CStr::from_ptr(c_buf);
            if let Ok(str_slice) = c_str.to_str() {
                token_stream.push(id);
                print!("{}", str_slice);
                std::io::stdout().flush().unwrap();
            }
        }

        println!("\n");
    }
}

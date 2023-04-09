# Example

Full example of using llama-rs.

Put your models in the appropriate folder, eg.
    
    ../models

Then run:

    cargo run --release

You should see something like:

```
    Finished release [optimized] target(s) in 0.17s
     Running `target/release/example`

llama_model_load: loading model from '../models/13B/model.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 5120
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 40
llama_model_load: n_layer = 40
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 13824
llama_model_load: n_parts = 2
llama_model_load: type    = 2
llama_model_load: ggml map size = 7759.83 MB
llama_model_load: ggml ctx size = 101.25 KB
llama_model_load: mem required  = 9807.93 MB (+ 1608.00 MB per state)
llama_model_load: loading tensors from '../models/13B/model.bin'
llama_model_load: model size =  7759.39 MB / num tensors = 363
llama_init_from_file: kv self size  =  400.00 MB

bob is a space pilot. Alice is a potato. This is a conversation between bob and alice:
Bob: "Hello, my name is Bob."
Alice: "Hello, I'm Alice."
Bob: "What are you doing today?"
Alice: "I am working on a potato farm."
target
Bob: "I am flying a spaceship.": _The best time to plant a tree was 20 years ago; the second best time is now._
– CHINESE PROVERB
**O** ne of my favorite parts about farming is that there are so many things you can do, and so many different ways to farm, and so many different crops. It's all up to you! You get to choose what you grow, how much space you have, and where you want to sell your products.
The problem is, when I was just getting started with my first few acres of farming, there wasn't a whole lot of information on these different methods of growing crops available. There weren't any books on the market that described how I could grow on less land, with fewer inputs, and still make a profit. Not only did I have to figure out all of this new stuff, but I also had to learn how to write the business plan that
```
Usefull commands to run concept decomposition for a given token of interest, a model, and specific layer

### Llava 1.5
CUDA_VISIBLE_DEVICES=0 python src/save_features.py --model_name llava-hf/llava-1.5-7b-hf  \
    --dataset_name WikiArtDataset  \
    --data_dir /mnt/swordfish-datastore/amith/corpora/wikiart \
    --annotation_file '' \
    --prompt_template llava \
    --hook_name save_hidden_states_for_token_of_interest \
    --modules_to_hook language_model.model.norm,language_model.model.layers.5,language_model.model.layers.15,language_model.model.layers.30 \
    --select_token_of_interest_samples \
    --token_of_interest Impressionism \
    --save_dir /local/nlp/milad/code/xl-vlms/data/wiki-art/impress  \
    --save_filename llava-hf  \
    --generation_mode \
    --exact_match_modules_to_hook \
    --save_only_generated_tokens \
    --dataset_size 500


python src/analyse_features.py --model_name llava-hf/llava-1.5-7b-hf \
  --analysis_name decompose_activations_text_grounding_image_grounding \
  --features_path /local/nlp/milad/code/xl-vlms/data/wiki-art/impress/features/save_hidden_states_for_token_of_interest_llava-hf.pth \
  --module_to_decompose language_model.model.layers.30 \
  --num_concepts 10 \
  --token_of_interest Impressionism \
  --num_grounded_text_tokens 20 \
  --decomposition_method snmf \
  --save_filename results_layers_30 \
  --save_dir /local/nlp/milad/code/xl-vlms/data/wiki-art/impress/

  ### Molmo

  CUDA_VISIBLE_DEVICES=0 python src/save_features.py --model_name allenai/Molmo-7B-O-0924  \
    --dataset_name WikiArtDataset  \
    --data_dir /mnt/swordfish-datastore/amith/corpora/wikiart \
    --annotation_file '' \
    --prompt_template llava \
    --hook_name save_hidden_states_for_token_of_interest \
    --modules_to_hook language_model.model.norm,language_model.model.layers.5,language_model.model.layers.15,language_model.model.layers.30 \
    --select_token_of_interest_samples \
    --token_of_interest Impressionism \
    --save_dir /local/nlp/milad/code/xl-vlms/data/wiki-art/molmo/impress  \
    --save_filename llava-hf  \
    --generation_mode \
    --exact_match_modules_to_hook \
    --save_only_generated_tokens \
    --dataset_size 100

python src/analyse_features.py --model_name allenai/Molmo-7B-O-0924 \
  --analysis_name decompose_activations_text_grounding_image_grounding \
  --features_path /local/nlp/milad/code/xl-vlms/data/wiki-art/molmo/impress/features/save_hidden_states_for_token_of_interest_llava-hf.pth \
  --module_to_decompose language_model.model.layers.30 \
  --num_concepts 5 \
  --token_of_interest Impressionism \
  --num_grounded_text_tokens 20 \
  --decomposition_method snmf \
  --save_filename results_layers_30 \
  --save_dir /local/nlp/milad/code/xl-vlms/data/wiki-art/molmo/impress/


  ### GalleryGPT

  CUDA_VISIBLE_DEVICES=0 python src/save_features.py --model_name /mnt/swordfish-pool2/milad/explainability-for-art-images/vllm-models/GalleryGPT/   \
    --dataset_name WikiArtDataset  \
    --data_dir /mnt/swordfish-datastore/amith/corpora/wikiart \
    --annotation_file '' \
    --prompt_template llava_v0 \
    --hook_name save_hidden_states_for_token_of_interest \
    --modules_to_hook model.layers.5,model.layers.15,model.model.layers.30 \
    --select_token_of_interest_samples \
    --token_of_interest Impressionism \
    --save_dir /local/nlp/milad/code/xl-vlms/data/wiki-art/gallery-gpt/impress  \
    --save_filename llava-hf  \
    --generation_mode \
    --exact_match_modules_to_hook \
    --save_only_generated_tokens \
    --dataset_size 500

python src/analyse_features.py --model_name /mnt/swordfish-pool2/milad/explainability-for-art-images/vllm-models/GalleryGPT/   \
  --analysis_name decompose_activations_text_grounding_image_grounding \
  --features_path /local/nlp/milad/code/xl-vlms/data/wiki-art/gallery-gpt/impress/features/save_hidden_states_for_token_of_interest_llava-hf.pth \
  --module_to_decompose model.layers.30 \
  --num_concepts 10 \
  --token_of_interest Impressionism \
  --num_grounded_text_tokens 20 \
  --decomposition_method snmf \
  --save_filename results_layers_30 \
  --save_dir /local/nlp/milad/code/xl-vlms/data/wiki-art/gallery-gpt/impress/


## Causal analysis

python src/analyse_features.py --model_name llava-hf/llava-1.5-7b-hf \
   --dataset_name WikiArtDataset  \
   --data_dir /mnt/swordfish-datastore/amith/corpora/wikiart \
   --analysis_name causal_analysis \
   --analysis_saving_path /local/nlp/milad/code/xl-vlms/data/wiki-art/impress/decompose_activations_text_grounding_image_grounding_results_layers_30.pth \
   --features_path /local/nlp/milad/code/xl-vlms/data/wiki-art/impress/features/save_hidden_states_for_token_of_interest_llava-hf.pth \
   --module_to_decompose language_model.model.layers.30 \
   --token_of_interest Impressionism \
   --save_filename causal_analysis_results_layers_30 \
   --save_dir /mnt/swordfish-pool2/milad/code/xl-vlms/data/wiki-art/gallery-gpt/impress/ \
   --data_size 500
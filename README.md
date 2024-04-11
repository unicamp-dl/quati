# Quati Information Retrieval Dataset

Quati dataset aims to support Brazilian Portuguese (pt-br) Information Retrieval (IR) systems development, providing document passagens originally created in pt-br, as well as queries (topics) created by native speakers.

Quati dataset is currently released in two versions: one with 1 million passages, and a larger one with 10 million passages. So far we have prepared only validation qrels for both versions, annotating 50 topics with an average of 97.78 passages per query on the 10M version, and 38.66 passages per query on the 1M version.

Quati is publicly release in Hugging Face Hub, and can be found [here](https://huggingface.co/datasets/unicamp-dl/quati).

This repository contains the main scripts used to annotate the query-passages released for Brazilian Portuguese (pt-br) IR systems evaluation. Check [our publication](https://arxiv.org/abs/2404.06976) for further details about the dataset.


## Files description

* [`relevance.py`](relevance.py): Main script used to access an LLM API to request the passage relevance annotation for a given query. Currently implemented for OpenAI's ChatGPT-3.5 and ChatGPT-4 models (openai library version 0.27.8). Need to be changed if targetting different LLMs.

* [`initial_prompt_rev_2.1.json`](initial_prompt_rev_2.1.json): JSON containing the query-passage annotation prompt, written in pt-br, to be used when accessing the target LLM. Contains the ChatGPT `chat/completions` format. Need to be changed if targetting different Languages.

* [`examples_rev_2.1.json`](examples_rev_2.1.json): Contains the few-shot examples composing our Chain-of-Thoughts prompt, also written in pt-br. Also need to be changed if targetting a different Language.

* [`config.json`](config.json): Contains configurable string formatting elements to help composing the prompt (written in pt-br) and parsing the LLM result (focusing JSON-encapsulated responses). Also need to be changed if targetting a different Language.

* `qrels`: contains the Quati released qrels, for both the 1M and 10M dataset versions.

* `human_annotations`: holds the human annotations, performed for 24 selected topics, used to evaluate the LLM annotation performance. Check [our publication](https://arxiv.org/abs/2404.06976) for details.


## Example of use

### Requesting LLM annotation for a set of retrieved query-passages:

```shell
python3 relevance.py --model "gpt-4-1106-preview" --initial_prompt "initial_prompt_rev_2.1.json" --examples "examples_rev_2.1.json" --config "config.json" --verbose True --history "<accumulated-annotated_query_passages_file>" --api_keys "<your-LLM-API-keys>.json" --api_key_to_use "<the-API-key-to-use-from-JSON>" --query_passages "<retrieved-query-passages-to-anotate>.tsv" --output "<LLM-query-passages-annotation>.tsv"
```

Where:

**history**: a file to register the query-passages already annotated, to avoid sending to LLM a query-passage combination it has already annotated ― which might happen if you are evaluating the retrieval results of different IR systems. Shall be used accross different `relevance.py` calls.

**query_passages**: a tab-separated file, containing at least the following fields: *query*, *passage_id*, *passage*, respectivelly the query text, the passage identification ― to be saved in the history ― and the passage text itself. One example is the following:

        query_id	query	passage_id	colbertx_id	position	score	passage
        1	Qual a maior característica da fauna brasileira?	clueweb22-pt0001-01-18874_0	4875435	1	0.41935885	Conheça 11 lindos animais típicos da fauna brasileira / Incrível Conheça 11 lindos animais típicos da fauna brasileira 4--4 Compartilhe no Facebook Compartilhe no Facebook Compartilhe no Pinterest Que o Brasil é gigante, todos nós sabemos e até já demos aqui exemplos chocantes sobre isso. Mas um outro exemplo dessa grandiosidade é quando sabemos que o país abriga 17% das espécies de aves e 10% das de anfíbios e mamíferos de todo o mundo. Neste post, o Incrível.club vai te mostrar uma lista de animais típicos do país. Mico-leão dourado © Nadine Doerle / Pixabay Este macaco “ruivo” ficou famoso ao tornar-se um símbolo de espécies brasileiras em extinção. O cuidado para a preservação da espécie, que é nativa da Mata Atlântica e aparece na nota de 20 reais, começou nos anos 1970 e se intensificou duas décadas depois. Embora a situação tenha melhorado, o risco de extinção ainda existe. Tamanduá-bandeira © joelfotos / Pixabay Este gigante que ama comer formigas — pode devorar até 30 mil por dia, inclusive cupins — é também um forte: com seus 2,20 metros, consegue até brigar com uma onça pintada. E vencer. Embora ele exista originalmente nas três Américas, já foi extinto em países como Estados Unidos e em Estados brasileiros como Espírito Santo e Rio e Janeiro. Onça pintada
        
**output**: output file name to hold the LLM annotations results; it will be a tab-separated file, as the sample below:

        query_id	query	passage_id	colbertx_id	position	score	passage	score	reasoning	usage	cost	duration	saved_cost
        1	Qual a maior característica da fauna brasileira?	clueweb22-pt0001-01-18874_0	4875435	1	0.41935885	Conheça 11 lindos animais típicos da fauna brasileira / Incrível Conheça 11 lindos animais típicos da fauna brasileira 4--4 Compartilhe no Facebook Compartilhe no Facebook Compartilhe no Pinterest Que o Brasil é gigante, todos nós sabemos e até já demos aqui exemplos chocantes sobre isso. Mas um outro exemplo dessa grandiosidade é quando sabemos que o país abriga 17% das espécies de aves e 10% das de anfíbios e mamíferos de todo o mundo. Neste post, o Incrível.club vai te mostrar uma lista de animais típicos do país. Mico-leão dourado © Nadine Doerle / Pixabay Este macaco “ruivo” ficou famoso ao tornar-se um símbolo de espécies brasileiras em extinção. O cuidado para a preservação da espécie, que é nativa da Mata Atlântica e aparece na nota de 20 reais, começou nos anos 1970 e se intensificou duas décadas depois. Embora a situação tenha melhorado, o risco de extinção ainda existe. Tamanduá-bandeira © joelfotos / Pixabay Este gigante que ama comer formigas — pode devorar até 30 mil por dia, inclusive cupins — é também um forte: com seus 2,20 metros, consegue até brigar com uma onça pintada. E vencer. Embora ele exista originalmente nas três Américas, já foi extinto em países como Estados Unidos e em Estados brasileiros como Espírito Santo e Rio e Janeiro. Onça pintada	2	A passagem menciona a diversidade da fauna brasileira, destacando que o país abriga 17% das espécies de aves e 10% das de anfíbios e mamíferos do mundo, o que pode ser considerado uma característica marcante. No entanto, a passagem não resume explicitamente qual é a 'maior característica', mas fornece dados que permitem inferir sobre a rica biodiversidade.	{'prompt_tokens': 869, 'completion_tokens': 109, 'total_tokens': 978}	0.0	6.921509742736816	0.03261
        
The output adds the **reasoning**, **usage**, **cost**, **duration**, and **saved_cost** fields to the input `query_passage` file.
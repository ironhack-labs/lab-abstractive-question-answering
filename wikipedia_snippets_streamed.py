"""Wikipedia snippets in parquet format"""
import math
import pyarrow.parquet as pq
import datasets

logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@ONLINE {wikidump,
    author = {Wikimedia Foundation},
    title  = {Wikimedia Downloads},
    url    = {https://dumps.wikimedia.org}
}
"""
_DESCRIPTION = """\
The dataset was built from the Wikipedia dump (https://dumps.wikimedia.org/).
Each example contains the content of one full Wikipedia article with cleaning to strip
markdown and unwanted sections (references, etc.).
"""
_LICENSE = (
    "This work is licensed under the Creative Commons Attribution-ShareAlike "
    "3.0 Unported License. To view a copy of this license, visit "
    "http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to "
    "Creative Commons, PO Box 1866, Mountain View, CA 94042, USA."
)


class WikipediaSnippetsStreamed(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "wiki_id": datasets.Value("string"),
                    "start_paragraph": datasets.Value("int32"),
                    "start_character": datasets.Value("int32"),
                    "end_paragraph": datasets.Value("int32"),
                    "end_character": datasets.Value("int32"),
                    "article_title": datasets.Value("string"),
                    "section_title": datasets.Value("string"),
                    "passage_text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://dumps.wikimedia.org",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        url = "https://storage.googleapis.com/huggingface-nlp/cache/datasets/wiki40b/en/1.1.0/wiki40b-train.parquet"
        downloaded_file = dl_manager.download(url)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_file}),
        ]

    def wiki40b_article_snippets(self, article, passage_len=100, overlap=0):
        paragraphs = article["text"].split("\n")
        aticle_idx = paragraphs.index("_START_ARTICLE_") + 1
        article_title = paragraphs[aticle_idx] if aticle_idx < len(paragraphs) else ""
        section_indices = [i + 1 for i, par in enumerate(paragraphs[:-1]) if par == "_START_SECTION_"]
        par_tabs = [par.split(" ") for par in paragraphs]
        word_map = [
            (i, len(" ".join(par[:j])), w)
            for i, par in enumerate(par_tabs)
            if not par[0].startswith("_START_")
            for j, w in enumerate(par)
            if i > 0
        ]
        step_size = passage_len - overlap
        passages = []
        for i in range(math.ceil(len(word_map) / step_size)):
            pre_toks = word_map[i * step_size: i * step_size + passage_len]
            start_section_id = max([0] + [j for j in section_indices if j <= pre_toks[0][0]])
            section_ids = [j for j in section_indices if j >= start_section_id and j <= pre_toks[-1][0]]
            section_ids = section_ids if len(section_ids) > 0 else [0]
            passage_text = " ".join([w for p_id, s_id, w in pre_toks])
            passages += [
                {
                    "article_title": article_title,
                    "section_title": " & ".join([paragraphs[j] for j in section_ids]),
                    "wiki_id": article["wikidata_id"],
                    "start_paragraph": pre_toks[0][0],
                    "start_character": pre_toks[0][1],
                    "end_paragraph": pre_toks[-1][0],
                    "end_character": pre_toks[-1][1] + len(pre_toks[-1][2]) + 1,
                    "passage_text": passage_text.replace("_NEWLINE_", "\n"),
                }
            ]
        return passages

    def _generate_examples(self, filepath):
        logger.info("generating examples from = %s", filepath)
        passage_counter = 0
        with open(filepath, "rb") as f:
            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                batch_table_dict = pf.read_row_group(i).to_pydict()
                for idx, article_text in enumerate(batch_table_dict["text"]):
                    article = {"text": article_text, "wikidata_id": batch_table_dict["wikidata_id"][idx]}
                    passages = self.wiki40b_article_snippets(article)
                    for passage in passages:
                        yield ++passage_counter, passage

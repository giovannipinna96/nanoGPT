"""Prepare a FineWeb-Edu subset in the nanoGPT .bin format.

Mirrors data/openwebtext/prepare.py (same GPT-2 BPE, same uint16 memmap layout) but
downloads only a few parquet shards of the `sample-10BT` subset instead of the 54 GB of
OpenWebText: this project only needs ~1B tokens for the grid, and the
point of the run is to demonstrate the mechanism, not to chase a headline number.

Two phases, because on this cluster the login node has network access and the compute
nodes are the ones with CPUs to spare:

    uv run python data/finewebedu/prepare.py --download-only     # login node, network
    uv run python data/finewebedu/prepare.py                     # compute node, CPU

The .bin files are written under $HYBRID_ATTN_DATA (i.e. /share) because the home
filesystem has no free inodes; symlinks are created next to this file so that the
unmodified train.py, which looks for data/<dataset>/train.bin, still finds them.

It also reports the MEDIAN DOCUMENT LENGTH in tokens. That number is needed to read the
results honestly: if the median document is shorter than
the sliding window W, the local layers already see whole documents and SWA loses nothing
-- a null degradation that would NOT generalise to genuinely long contexts.
"""

import argparse
import os

import numpy as np
import tiktoken
from tqdm import tqdm

REPO = "HuggingFaceFW/fineweb-edu"
SHARD_FMT = "sample/10BT/{:03d}_00000.parquet"   # 14 shards, ~714M gpt2 tokens each
HERE = os.path.dirname(os.path.abspath(__file__))

enc = tiktoken.get_encoding("gpt2")


def out_dir():
    base = os.environ.get("HYBRID_ATTN_DATA", HERE)
    d = os.path.join(base, "finewebedu") if base != HERE else HERE
    os.makedirs(d, exist_ok=True)
    return d


def download(n_shards):
    from huggingface_hub import hf_hub_download
    paths = []
    for i in range(n_shards):
        f = SHARD_FMT.format(i)
        print(f"downloading {f} ...", flush=True)
        paths.append(hf_hub_download(REPO, f, repo_type="dataset"))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=2,
                    help="parquet shards of sample-10BT (~714M gpt2 tokens each)")
    ap.add_argument("--val-fraction", type=float, default=0.002,
                    help="held-out fraction of documents (~2.9M tokens with 2 shards)")
    ap.add_argument("--num-proc", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    paths = download(args.shards)
    if args.download_only:
        print("download done:", *paths, sep="\n  ")
        return

    from datasets import load_dataset
    dataset = load_dataset("parquet", data_files={"train": paths}, split="train",
                           num_proc=args.num_proc)
    print(dataset)

    # seed 2357 is the value used by nanoGPT's openwebtext prepare.py
    split = dataset.train_test_split(test_size=args.val_fraction, seed=2357, shuffle=True)
    split["val"] = split.pop("test")

    def process(example):
        ids = enc.encode_ordinary(example["text"])  # ignores special tokens
        ids.append(enc.eot_token)                   # 50256 for gpt2 bpe
        return {"ids": ids, "len": len(ids)}

    tokenized = split.map(process, remove_columns=dataset.column_names,
                          desc="tokenizing the splits", num_proc=args.num_proc)

    dst = out_dir()
    stats = {}
    for name, dset in tokenized.items():
        lens = np.array(dset["len"], dtype=np.int64)
        arr_len = lens.sum(dtype=np.uint64)
        filename = os.path.join(dst, f"{name}.bin")
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for b in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=b,
                               contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        stats[name] = dict(docs=len(lens), tokens=int(arr_len),
                           median_doc_tokens=int(np.median(lens)),
                           p90_doc_tokens=int(np.percentile(lens, 90)))
        # symlink so that the unmodified train.py finds data/<dataset>/<split>.bin
        link = os.path.join(HERE, f"{name}.bin")
        if os.path.realpath(link) != os.path.realpath(filename):
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(filename, link)

    print("\n== dataset summary ==")
    for name, s in stats.items():
        print(f"{name}: {s['docs']:,} docs, {s['tokens']:,} tokens, "
              f"median doc {s['median_doc_tokens']} tokens, p90 {s['p90_doc_tokens']}")
    print("compare the median against the sliding window W=256: if the median document is "
          "shorter than W, local layers already see whole documents and SWA loses little "
          "by construction.")
    with open(os.path.join(dst, "summary.txt"), "w") as fh:
        for name, s in stats.items():
            fh.write(f"{name}\t{s['docs']}\t{s['tokens']}\t{s['median_doc_tokens']}\t"
                     f"{s['p90_doc_tokens']}\n")


if __name__ == "__main__":
    main()

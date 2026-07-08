import argparse
from collections import Counter
from functools import partial
import hashlib
from itertools import chain
import numpy as np
from pathlib import Path
import pickle
import re
from tqdm import tqdm
from typing import Generator, Iterable, List, Tuple
from typing_extensions import override
import sys
from .._cli import CliSubCommand, subcommand

__doc__ = "Data tools for Deepbio Toolkit"

class SequenceFilter:
    def __init__(self, show_progress: bool):
        self.sequence_index = {}
        self.sequence_count = Counter()
        self.identifier_map = {}
        self.show_progress = show_progress

    def pipe(self, sequence_stream: Iterable[Tuple[str, str]]) -> Generator[str, None, None]:
        if self.show_progress:
            sequence_stream = tqdm(sequence_stream, desc="Analyzing sequences")
        for identifier, sequence in sequence_stream:
            sequence_hash = hashlib.sha256(sequence.encode()).digest()
            seen = sequence_hash in self.sequence_index
            if not seen:
                self.sequence_index[sequence_hash] = len(self.sequence_index)
            self.sequence_count[sequence_hash] += 1
            self.identifier_map[identifier] = self.sequence_index[sequence_hash]
            if not seen:
                yield sequence


@subcommand("import_sequences", "Generate a random run ID.")
class ImportSequences(CliSubCommand):
    @override
    def configure(self, parser):
        parser.add_argument("input_paths", nargs="+", type=Path, help="Input FASTA(s) or FASTQ file(s) containing DNA sequences.")
        parser.add_argument("--output_id_map", "-i", action="store_true", help="Do not output an ID map.")
        parser.add_argument("--output_abundance_map", "-a", action="store_true", help="Do not output an abundance map.")
        parser.add_argument("--output", "-O", type=Path, required=False, help="Path to the output .seq file.")
        parser.add_argument("--silent", "-s", action="store_true", help="Do not show progress bar.")

    def read_fasta(self, path: Path, prefix_sequence_ids: bool) -> Generator[Tuple[str, str], None, None]:
        import gzip
        prefix = ""
        if prefix_sequence_ids:
            prefix = path.name.rstrip(".gz").rstrip(".fasta") + "."
        open_fn = partial(gzip.open, mode="rt") if path.suffix == ".gz" else open
        with open_fn(path) as f:
            identifier = ""
            sequence = ""
            for line in f:
                if line.startswith(">"):
                    if sequence != "":
                        yield prefix+identifier, sequence
                    identifier = re.match(r">(\S+)", line).group(1) # type: ignore
                    sequence = ""
                else:
                    sequence += line.strip()
            if sequence != "":
                yield prefix+identifier, sequence

    def read_fastq(self, path: Path, prefix_sequence_ids: bool) -> Generator[Tuple[str, str], None, None]:
        import gzip
        prefix = ""
        if prefix_sequence_ids:
            # try to extract name from standard fastq format
            match = re.match(r"(.+)_S\d+_L\d+_R\d+_\d+.fastq(?:\.gz)?", path.name)
            if match is not None:
                prefix = match.group(1) + "."
            else:
                prefix = path.name.rstrip(".gz").rstrip(".fastq") + "."
        open_fn = partial(gzip.open, mode="rt") if path.suffix == ".gz" else open
        with open_fn(path) as f:
            prev = ""
            index = 0
            for line in f:
                if line.startswith("+"):
                    yield f"{prefix}{index}", prev.rstrip()
                prev = line

    def read(self, path: Path, prefix_sequence_ids: bool) -> Generator[Tuple[str, str], None, None]:
        if path.name.rstrip(".gz").endswith(".fasta"):
            return self.read_fasta(path, prefix_sequence_ids)
        if path.name.rstrip(".gz").endswith(".fastq"):
            return self.read_fastq(path, prefix_sequence_ids)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def write(self, config):
        from dbtk.data.formats import SequenceStore
        sequence_filter = SequenceFilter(not config.silent)
        sequence_stream = chain.from_iterable(
            self.read(path, prefix_sequence_ids=(len(config.input_paths) > 1))
            for path in config.input_paths
        )
        SequenceStore.create(
            sequence_filter.pipe(sequence_stream),
            config.output.with_suffix(".seq"),
            show_progress=(not config.silent)
        )
        if not config.silent:
            print("Finishing up...")
        if config.output_id_map:
            with open(config.output.with_suffix(".idmap"), "wb") as f:
                print("Writing", len(sequence_filter.identifier_map), "identifiers to ID map.")
                pickle.dump(sequence_filter.identifier_map, f)
        if config.output_abundance_map:
            with open(config.output.with_suffix(".abundance"), "wb") as f:
                unit = max(int(2**np.ceil(np.log2(np.ceil(np.log2(max(sequence_filter.sequence_count.values()))) + 1e-8))), 8)
                index_dtype = getattr(np, f"uint{unit}")
                counts = np.array(list(sequence_filter.sequence_count.values()), dtype=index_dtype)
                f.write(np.uint8(unit).tobytes() + counts.tobytes())
        if not config.silent:
            print("Analyzed", sum(sequence_filter.sequence_count.values()), "sequences.")
            print("# Unique sequences:", len(sequence_filter.sequence_index))
            print("# Duplicate sequences:", sum(count for count in sequence_filter.sequence_count.values() if count > 1))
            print(f"{1-len(sequence_filter.sequence_index)/sum(sequence_filter.sequence_count.values()):.2%}% sequence reduction")
            print("")

            # print total file size reduction (including abundance and id map if present)
            total_input_size = sum(path.stat().st_size for path in config.input_paths)
            total_output_size = config.output.with_suffix(".seq").stat().st_size
            percent_difference = 1 - total_output_size/total_input_size
            if config.output_id_map:
                total_output_size += config.output.with_suffix(".idmap").stat().st_size
            if config.output_abundance_map:
                total_output_size += config.output.with_suffix(".abundance").stat().st_size
            # determine appropriate units
            units = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            while total_input_size > 1024 and i < len(units):
                total_input_size /= 1024
                i += 1
            print(f"Total input size: {total_input_size:.2f} {units[i]}")
            i = 0
            while total_output_size > 1024 and i < len(units):
                total_output_size /= 1024
                i += 1
            print(f"Total output size: {total_output_size:.2f} {units[i]}")
            print(f"{percent_difference:.2%}% file size reduction")

            print("Done.")

    def run_multiple(self, config: argparse.Namespace) -> int:
        if config.output is None:
            print("Output path is required when importing multiple files.", file=sys.stderr)
            return 1
        if config.output.is_dir():
            print("Output path must be a file when importing multiple files.", file=sys.stderr)
            return 1
        self.write(config)
        return 0

    def run_single(self, config: argparse.Namespace) -> int:
        input_path = config.input_paths[0]
        input_file_name = re.sub(r"\.(fastq|fasta)(\.gz)?$", "", input_path.name)
        if config.output is None:
            # strip fasta/fastq + .gz extension
            config.output = input_path.with_name(input_file_name)
        elif config.output.is_dir():
            config.output = config.output / input_file_name
        self.write(config)
        return 0

    def run(self, config: argparse.Namespace) -> int:
        """
        Import the provided FASTA/FASTQ files into a sequence store.

        Args:
            config (argparse.Namespace): _description_

        Returns:
            int: _description_
        """
        if len(config.input_paths) > 1:
            return self.run_multiple(config)
        return self.run_single(config)


@subcommand("generate_mappings", "Generate reference-based sample mapping databases for SetBERT pretraining.")
class GenerateMappings(CliSubCommand):
    _CONFIG_SCHEMA = """\
Config schema (YAML):
    classifier:
      name: topdown-64d-250bp-1mer   # used in output filename; must match the name used in 'dbtk model predict'

    reference_dataset:
      name: silva-nr99-filtered-515f-806r
      path: /path/to/datasets/silva_nr99_filtered_515f_806r

    datasets:
      - name: my-dataset
        path: /path/to/datasets/my-dataset

    seed: 42
    overwrite: false

Reads per dataset:
    {dataset.path}/predictions.{classifier.name}.pt   (produced by 'dbtk model predict')

Output per dataset:
    {dataset.path}/sequences.{classifier.name}.{reference_dataset.name}.fasta.mapping.db
"""

    @override
    def configure(self, parser: argparse.ArgumentParser):
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = self._CONFIG_SCHEMA
        parser.add_argument("--config", type=Path, required=True, help="YAML config file.")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mapping files.")
        parser.add_argument("--seed", type=int, default=None, help="Override random seed from config.")

    def _build_genus_index(self, tax_db) -> list:
        """Return a list where index i holds all reference sequence indices for genus i (alpha/taxon_id order)."""
        n_genera = len(tax_db.tree.id_to_taxon_map[-1])
        genus_index = [[] for _ in range(n_genera)]
        for node in tax_db.tree.taxonomy_id_map[-1]:
            seqs = tax_db.sequence_indices_with_taxonomy_id(node.taxonomy_id)
            genus_index[node.taxon_id].extend(seqs)
        return genus_index

    def _build_mapping(self, genus_preds, genus_index, dataset_path, output_path, ref_fasta_db, fasta_db, rng):
        """Build the LMDB reference mapping from pre-computed genus predictions (rank-0 only)."""
        from dnadb import fasta

        real_samples = fasta_db.mappings(dataset_path / "sequences.fasta.mapping.db", load_into_memory=True)
        print(f"  {len(real_samples)} samples, {len(fasta_db):,} unique sequences")

        empty = sum(1 for gid in genus_preds if len(genus_index[gid]) == 0)
        if empty:
            print(f"  Warning: {empty:,} sequences predict a genus with no reference entries (reads will be dropped)")

        mapping_factory = fasta.FastaMappingDbFactory(output_path, ref_fasta_db)
        for sample in tqdm(real_samples, desc="  Building reference mappings"):
            entry = mapping_factory.create_entry(sample.name)
            raw_indices = np.array([sample.sequence_index(i) for i in range(len(sample))], dtype=np.int64)
            unique_indices, counts = np.unique(raw_indices, return_counts=True)
            for real_idx, count in zip(unique_indices, counts):
                candidates = genus_index[int(genus_preds[real_idx])]
                if len(candidates) == 0:
                    continue
                chosen = rng.choice(candidates, size=int(count), replace=True)
                for ref_idx in chosen:
                    entry.write_sequence_index(int(ref_idx))
            mapping_factory.write_entry(entry)
        mapping_factory.close()

    @override
    def run(self, args: argparse.Namespace) -> int:
        import torch
        from dnadb import fasta, taxonomy
        from dbtk.config import load as load_config

        cfg = load_config(args.config)

        classifier = cfg["classifier"]
        name = classifier["name"]
        ref_name = cfg["reference_dataset"]["name"]
        ref_path = Path(cfg["reference_dataset"]["path"])
        datasets = [(d["name"], Path(d["path"])) for d in cfg["datasets"]]
        seed = args.seed if args.seed is not None else cfg.get("seed", 42)
        overwrite = args.overwrite or cfg.get("overwrite", False)

        print(f"Classifier: {name}")
        print(f"Loading reference from {ref_path}...")
        ref_fasta_db = fasta.FastaDb(ref_path / "sequences.fasta.db")
        ref_tax_db = taxonomy.TaxonomyDb(
            ref_path / "taxonomy.tax.db",
            in_memory=taxonomy.TaxonomyDb.InMemory.SequencesWithTaxonomy,
        )
        print(f"  {len(ref_fasta_db):,} sequences, {ref_tax_db.num_labels:,} genera")

        print("Building genus → reference sequence index...")
        genus_index = self._build_genus_index(ref_tax_db)

        rng = np.random.default_rng(seed)

        for dataset_name, dataset_path in datasets:
            output_name = f"sequences.{name}.{ref_name}.fasta.mapping.db"
            output_path = dataset_path / output_name

            if output_path.exists() and not overwrite:
                print(f"\n[{dataset_name}] Already exists, skipping (use --overwrite to redo).")
                continue

            fasta_db_path = dataset_path / "sequences.fasta.db"
            mapping_db_path = dataset_path / "sequences.fasta.mapping.db"
            pred_path = dataset_path / f"predictions.{name}.pt"

            if not fasta_db_path.exists() or not mapping_db_path.exists():
                print(f"\n[{dataset_name}] Skipping: sequences.fasta.db or .mapping.db not found.")
                continue

            if not pred_path.exists():
                print(f"\n[{dataset_name}] Skipping: {pred_path.name} not found. "
                      f"Run 'dbtk model predict' first.")
                continue

            print(f"\n[{dataset_name}] Loading predictions from {pred_path.name}...")
            payload = torch.load(pred_path, weights_only=False)
            all_seq_ids   = payload["seq_ids"]
            all_genus_ids = payload["pred_ids"][:, -1]  # genus = last rank

            # Reindex predictions by sequence position in the FASTA DB
            fasta_db = fasta.FastaDb(fasta_db_path)
            n_sequences = len(fasta_db)
            seq_id_to_idx = {entry.identifier: i for i, entry in enumerate(fasta_db)}
            genus_preds = np.empty(n_sequences, dtype=np.int64)
            for seq_id, pred in zip(all_seq_ids, all_genus_ids.numpy()):
                genus_preds[seq_id_to_idx[seq_id]] = pred

            print(f"[{dataset_name}] Generating reference mapping...")
            self._build_mapping(
                genus_preds=genus_preds,
                genus_index=genus_index,
                dataset_path=dataset_path,
                output_path=output_path,
                ref_fasta_db=ref_fasta_db,
                fasta_db=fasta_db,
                rng=rng,
            )
            fasta_db.close()
            print(f"[{dataset_name}] Written to {output_path}")

        print("\nAll done.")
        return 0

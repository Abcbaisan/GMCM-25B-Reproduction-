"""Small, independent checks of parsing, physical units, filtering and caching."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from openpyxl import Workbook

from gmcm25b.data import (CSI_COLUMNS, channel_to_sinr, load_dataset, parse_csi,
                         prepare_dataset, read_workbook)


def write_workbook(path, split="train", cases=None):
    book = Workbook()
    sheet = book.active
    header = [None, *CSI_COLUMNS]
    if split == "train":
        header.append("mcs")
    header += ["dfx_time", "csi_time", "beamforming_en", "noise_floor"]
    sheet.append(header)
    for index, updates in enumerate(cases or [{}]):
        values = {name: str([complex(1e-5, 1e-5)] * 122) for name in CSI_COLUMNS}
        values.update(mcs=275.3, dfx_time=1.25, csi_time=1.5,
                      beamforming_en=0, noise_floor=-100)
        values.update(updates)
        sheet.append([index, *[values[name] for name in header[1:]]])
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    book.close()


class CsiTests(unittest.TestCase):
    def test_parse_literals(self):
        vector = parse_csi("[(1+2j), (-3-4j), 5e-3j]", expected_size=3)
        np.testing.assert_array_equal(vector, [1 + 2j, -3 - 4j, .005j])

    def test_reject_malformed_shape_nonfinite_and_expressions(self):
        for value in (None, "[1j]", "[" + ",".join(["nan"] * 122) + "]",
                      "[" + ",".join(["1+2"] * 122) + "]",
                      "__import__('os').getcwd()", "(1+2j)"):
            with self.subTest(value=str(value)[:50]), self.assertRaises(ValueError):
                parse_csi(value)

    def test_eight_channels_and_power_conversion(self):
        channel = np.full((2, 4, 122), 1e-5 + 1e-5j)
        expected = (.001 * 8 * 2e-10) / (1e-13 / 122)
        actual = channel_to_sinr(channel, -100)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(channel_to_sinr(channel, -100, signal_scale=1), actual * 1000)
        expected_epsilon = (.001 * 8 * 2e-10) / (1e-13 / 122 + 1e-15)
        np.testing.assert_allclose(channel_to_sinr(channel, -100, noise_epsilon_w=1e-15), expected_epsilon)
        with self.assertRaises(ValueError):
            channel_to_sinr(channel[:1], -100)
        with self.assertRaises(ValueError):
            channel_to_sinr(channel, -130)
        with self.assertRaises(ValueError):
            channel_to_sinr(channel, -100, noise_epsilon_w=-1)


class WorkbookTests(unittest.TestCase):
    def test_training_filter_and_validation_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train_path, valid_path = root / "train.xlsx", root / "valid.xlsx"
            cases = [{}, {"noise_floor": -130}, {"beamforming_en": 1},
                     {CSI_COLUMNS[0]: "[1j]"}, {"mcs": None}]
            write_workbook(train_path, cases=cases)
            write_workbook(valid_path, split="valid", cases=cases)
            training = read_workbook(train_path, split="train", source_file="device/train.xlsx", device="device")
            validation = read_workbook(valid_path, split="valid", source_file="device/valid.xlsx", device="device")
            train, valid, audit = prepare_dataset([training, validation])
            self.assertEqual(len(train), 1)
            self.assertEqual(train.y.tolist(), [275.3])
            self.assertEqual(audit["train"]["rows_dropped"], 4)
            self.assertEqual(audit["train"]["reason_counts"]["invalid_mcs"], 1)
            self.assertEqual(len(valid), 5)
            self.assertEqual(valid.meta.excel_row.tolist(), [2, 3, 4, 5, 6])
            self.assertEqual(valid.meta.source_index.tolist(), ["0", "1", "2", "3", "4"])
            self.assertEqual(valid.meta.sample_id.iloc[2], "device/valid.xlsx::4")
            self.assertTrue(np.isnan(valid.sinr[1:4]).all())
            self.assertTrue(np.isfinite(valid.sinr[[0, 4]]).all())
            self.assertEqual(audit["valid"]["rows_dropped"], 0)

    def test_all_eight_csi_columns_required(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.xlsx"
            book = Workbook()
            book.active.append([*CSI_COLUMNS[:-1], "beamforming_en", "noise_floor", "mcs"])
            book.save(path)
            book.close()
            with self.assertRaisesRegex(ValueError, "csi_matrix_r1_c3"):
                read_workbook(path, split="train")

    def test_cache_bound_to_content_and_units(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            train_path = raw / "device/train/a.xlsx"
            valid_path = raw / "device/valid/a.xlsx"
            write_workbook(train_path)
            write_workbook(valid_path, split="valid", cases=[{}, {"noise_floor": -130}])
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": [
                {"path": "device/train/a.xlsx", "device": "device", "split": "train"},
                {"path": "device/valid/a.xlsx", "device": "device", "split": "valid"},
            ]}), encoding="utf-8")
            cache = root / "cache"
            train, valid, audit = load_dataset(raw, manifest, cache)
            cached_train, cached_valid, cached_audit = load_dataset(raw, manifest, cache)
            np.testing.assert_array_equal(train.sinr, cached_train.sinr)
            np.testing.assert_array_equal(valid.sinr, cached_valid.sinr)
            self.assertEqual(valid.meta.source_index.tolist(), cached_valid.meta.source_index.tolist())
            self.assertEqual(audit["sources"], cached_audit["sources"])
            self.assertTrue(Path(audit["row_audit_csv"]).is_file())
            scaled, _, _ = load_dataset(raw, manifest, cache, signal_scale=1)
            np.testing.assert_allclose(scaled.sinr, train.sinr * 1000)
            write_workbook(train_path, cases=[{"mcs": 300.1}])
            changed, _, changed_audit = load_dataset(raw, manifest, cache)
            self.assertEqual(changed.y.tolist(), [300.1])
            self.assertNotEqual(audit["sources"][0]["sha256"], changed_audit["sources"][0]["sha256"])


if __name__ == "__main__":
    unittest.main()

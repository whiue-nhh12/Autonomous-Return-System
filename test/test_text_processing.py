import unittest

from rag.core.utils.text_processing import decode_and_normalize, normalize_text


class TestTextProcessing(unittest.TestCase):
    def test_decodes_utf8_bom(self) -> None:
        value = bytes.fromhex("ef bb bf 54 69 e1 ba bf 6e 67 20 56 69 e1 bb 87 74")
        self.assertEqual(decode_and_normalize(value), "Tiếng Việt")

    def test_falls_back_to_vietnamese_code_page(self) -> None:
        self.assertEqual(decode_and_normalize(bytes.fromhex("43 61 66 e9")), "Café")

    def test_normalizes_nfc_and_removes_invisible_marks(self) -> None:
        self.assertEqual(normalize_text("Cafe\u0301\u200b\ufeff"), "Café")

    def test_normalizes_vietnamese_tone_mark_placement(self) -> None:
        value = "khỏan, HỌAT ĐỘNG, Tóan, hòan, ỦY BAN"
        self.assertEqual(normalize_text(value), "khoản, HOẠT ĐỘNG, Toán, hoàn, UỶ BAN")

    def test_keeps_document_identifiers_unchanged(self) -> None:
        value = "Điều 12/2024/TT-BTC, khỏan 1"
        self.assertEqual(normalize_text(value), "Điều 12/2024/TT-BTC, khoản 1")

    def test_repairs_mojibake_when_ftfy_is_available(self) -> None:
        try:
            import ftfy  # noqa: F401
        except ImportError:
            self.skipTest("ftfy is not installed")
        self.assertEqual(normalize_text("FranÃ§ais"), "Français")


if __name__ == "__main__":
    unittest.main()

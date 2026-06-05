import unittest
from automata.unseen_handler import UnseenHandler
import logging

logger = logging.getLogger(__name__)


class TestUnseenHandler(unittest.TestCase):

    def setUp(self):
        """her testten önce çalışacak hazırlık fonksiyonu veri oluşturma vesaire"""
        self.known_patterns = {"aabc", "abcc", "bcca"}
        self.handler = UnseenHandler(known_patterns=self.known_patterns)


    def test_levenshtein_distance_calculation(self):
        """levenshtein mesafe hesaplamasının matematiksel doğruluğunu test eder."""
        # birbirine aynı stringlerin arası 0 olması beklenir
        self.assertEqual(UnseenHandler.levenshtein_distance("aabc", "aabc"), 0)

        # 1 harf değişikliği arasındaki mesafe
        self.assertEqual(UnseenHandler.levenshtein_distance("bcca", "acca"), 1)

        # tamamen farklı stringler arası mesafe mantıken 4 olmalıdır
        self.assertEqual(UnseenHandler.levenshtein_distance("aaaa", "bbbb"), 4)

    def test_known_pattern_handling(self):
        """sözlükte olan bir pattern geldiğinde sistemin davranışını test et."""
        result = self.handler.handle("aabc")

        self.assertEqual(result["status"], "known")
        self.assertEqual(result["mapped_to"], "aabc")
        self.assertEqual(result["distance"], 0)

    def test_unseen_pattern_mapping(self):
        """sözlükte olmayan bir pattern geldiğinde en yakına doğru eşleniyor mu"""
        unseen_pattern = "abbc"
        result = self.handler.handle(unseen_pattern)

        self.assertEqual(result["status"], "unseen")
        self.assertEqual(result["distance"], 1)


        self.assertIn(result["mapped_to"], ["aabc", "abcc"])
    def test_empty_knowledge_base_raises_error(self):
        """boş bir sözlük verildiğinde ValueError fırlatılıyor mu test eder"""
        with self.assertRaises(ValueError):
            UnseenHandler(known_patterns=set())


if __name__ == "__main__":
    unittest.main()
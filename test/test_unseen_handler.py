import unittest
from automata.unseen_handler import UnseenHandler
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestUnseenHandler(unittest.TestCase):

    def setUp(self):
        """her testten önce çalışacak hazırlık fonksiyonu veri oluşturma vesaire"""
        self.known_patterns = {"aabc", "abcc", "bcca"}
        self.handler = UnseenHandler(known_patterns=self.known_patterns)

    def test_levenshtein_distance_calculation(self):
        """levenshtein mesafe hesaplamasının matematiksel doğruluğunu test eder"""
        # birbirine aynı stringlerin arası 0 olması beklenir
        self.assertEqual(UnseenHandler.levenshtein_distance("aabc", "aabc"), 0)

        # 1 harf değişikliği arasındaki mesafe
        self.assertEqual(UnseenHandler.levenshtein_distance("bcca", "acca"), 1)

        # tamamen farklı stringler arası mesafe mantıken 4 olmalıdır
        self.assertEqual(UnseenHandler.levenshtein_distance("aaaa", "bbbb"), 4)

    def test_known_pattern_handling(self):
        """sözlükte olan bir pattern geldiğinde sistemin davranışını test et"""
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

    def test_unseen_handler_logging_and_robustness(self):
        """sistem anomalileri ve unseen durumların log mekanizmasını doğrular"""
        logger.info("PR Kancası: Unseen handler kararlılık testi tetiklendi.")
        unseen_pattern = "xyzq"  # Tamamen alakasız gürültülü bir pattern

        result = self.handler.handle(unseen_pattern)

        # Log yapısının doğruluğunu ve geri dönen sözlük bütünlüğünü doğrula
        self.assertIn("status", result)
        self.assertIn("mapped_to", result)
        self.assertGreater(result["distance"], 0)

    def test_extreme_length_difference_handling(self):
        """farklı uzunluktaki kelimelerin mesafe sınırlarını test eder"""
        # Levenshtein kuralı gereği boş string ile 4 karakterli string arası mesafe karakter sayısı kadar olmalıdır
        distance_empty = UnseenHandler.levenshtein_distance("", "aabc")
        self.assertEqual(distance_empty, 4)

        # Çok uzun bir gürültü serisi geldiğinde sistemin patlamadığını test et
        distance_long = UnseenHandler.levenshtein_distance("aabc", "aabc_extra_noise_signals")
        self.assertEqual(distance_long, 20)

    def test_case_sensitivity_and_exact_match(self):
        """büyük-küçük harf duyarlılığını ve veri bütünlüğünü test eder"""
        # SAX alfabesi küçük harflerden oluştuğu için büyük harf gürültüsü gelirse mesafe üretilmelidir
        result = self.handler.handle("AABC")
        self.assertEqual(result["status"], "unseen")
        self.assertNotEqual(result["distance"], 0)


if __name__ == "__main__":
    unittest.main()
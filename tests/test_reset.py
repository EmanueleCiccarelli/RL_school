import unittest
from play2learn_lunarlander.reset import get_root_dir, verifica_integrita_progetto


class TestReset(unittest.TestCase):
    def test_integrity_check(self):
        root = get_root_dir()
        self.assertTrue(verifica_integrita_progetto(root))


if __name__ == "__main__":
    unittest.main()

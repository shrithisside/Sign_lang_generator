from django.test import TestCase, Client

class Voice2GestureTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_homepage_loads(self):
        """Test that the main web interface loads successfully (HTTP 200)."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_english_text_translation_api(self):
        """Test that English text translation returns valid JSON and sign cards."""
        response = self.client.post(
            '/translate_text/', 
            data={'text': 'HELLO', 'language': 'en'}, 
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get('success'))
        self.assertGreater(len(data.get('words', [])), 0)

    def test_hindi_text_translation_api(self):
        """Test that Hindi text (Devanagari) is translated to English and produces sign cards."""
        response = self.client.post(
            '/translate_text/', 
            data={'text': 'नमस्ते आप कैसे हैं', 'language': 'hi'}, 
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get('success'))
        self.assertEqual(data.get('source_language'), 'hi')
        self.assertIn('original_transcript', data)
        self.assertIn('translated_transcript', data)
        self.assertGreater(len(data.get('words', [])), 0)

    def test_marathi_text_translation_api(self):
        """Test that Marathi text is translated to English and produces sign cards."""
        response = self.client.post(
            '/translate_text/', 
            data={'text': 'नमस्कार तुम्ही कसे आहात', 'language': 'mr'}, 
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get('success'))
        self.assertEqual(data.get('source_language'), 'mr')
        self.assertIn('translated_transcript', data)
        self.assertGreater(len(data.get('words', [])), 0)

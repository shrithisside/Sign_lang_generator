from django.test import TestCase, Client

class Voice2GestureTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_homepage_loads(self):
        """Test that the main web interface loads successfully (HTTP 200)."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_text_translation_api(self):
        """Test that the text translation endpoint returns valid JSON."""
        response = self.client.post(
            '/translate_text/', 
            data={'text': 'HELLO'}, 
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json().get('success'))

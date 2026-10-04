import json
import unittest

from main import app


class ComparisonHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_returns_independent_engine_states(self):
        body = json.dumps({"reviews": [{
            "id": "r1", "text": "Great sound but late delivery", "star_rating": 3
        }]}).encode()
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/api/compare/lu-et-al",
            "raw_path": b"/api/compare/lu-et-al", "query_string": b"", "root_path": "",
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            "server": ("test", 80), "client": ("test", 0),
        }
        sent = []
        received = False

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        status = next(message["status"] for message in sent if message["type"] == "http.response.start")
        response = json.loads(b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body"))
        self.assertEqual(status, 200)
        self.assertEqual(response["taxonomy"][5], "seller_service")
        self.assertIn("authenticheck", response["engines"])
        self.assertIn("lu_et_al", response["engines"])
        self.assertEqual(response["engines"]["authenticheck"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()

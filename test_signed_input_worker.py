import io
import unittest
from email.message import Message
from unittest.mock import Mock

from signed_input_worker import fetch_image, resolve_input

URL = "https://ggggggg.uk/I2V/private/r2/i2v-client%2F" + "a" * 64 + ".png?token=test"


class SignedInputTests(unittest.TestCase):
    def test_original_is_first_and_existing_inline_payload_is_unchanged(self):
        face = {"kind": "image", "base64": "face"}
        data = {"task": "ref2va", "references": [face], "original_image_url": URL}
        self.assertEqual(resolve_input(data, lambda _, **kwargs: "original")["references"],
                         [{"kind": "image", "base64": "original"}, face])
        old = {"task": "i2va", "first_image_base64": "existing"}
        self.assertIs(resolve_input(old), old)
        self.assertEqual(resolve_input({"task": "i2va", "original_image_url": URL},
                                       lambda _, **kwargs: "original")["first_image_base64"], "original")

    def test_fetches_face_links_in_order_with_shared_byte_budget(self):
        first = URL
        second = URL.replace("a" * 64, "b" * 64)
        third = URL.replace("a" * 64, "c" * 64)
        fetch = Mock(side_effect=["YWJj", "ZA==", "ZWY="])
        payload = {"task": "ref2va", "original_image_url": first,
                   "references": [{"kind": "image", "url": second}, {"kind": "image", "url": third}]}
        result = resolve_input(payload, fetch)
        self.assertEqual([item["base64"] for item in result["references"]], ["YWJj", "ZA==", "ZWY="])
        self.assertEqual([call.args[0] for call in fetch.call_args_list], [first, second, third])
        limits = [call.kwargs["limit"] for call in fetch.call_args_list]
        self.assertEqual([limits[0] - n for n in limits], [0, 3, 4])

    def test_only_hub_image_urls_are_fetched(self):
        opener = Mock()
        with self.assertRaises(ValueError):
            fetch_image(URL.replace("ggggggg.uk", "example.com"), opener)
        opener.open.assert_not_called()

        class Response(io.BytesIO):
            status = 200
            headers = Message()
        Response.headers["Content-Type"] = "image/png"
        opener.open.return_value = Response(b"abc")
        self.assertEqual(fetch_image(URL, opener), "YWJj")


"""Integration tests for api.py FastAPI endpoints."""
import unittest
from fastapi.testclient import TestClient

from api import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("easings", data)

    def test_web_client_index(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("model-viewer", res.text)
        self.assertIn("video_monitor", res.text)

    def test_procedural_generation_endpoint(self):
        res = self.client.post("/api/generate/procedural", json={
            "n_frames": 16,
            "easing": "ease-in-out cubic",
            "edge_width": 2.0,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["shape"][0], 16)

    def test_video_endpoints(self):
        # 1. Test video frame retrieval
        res_frame = self.client.get("/api/video/frame/0")
        self.assertEqual(res_frame.status_code, 200)
        self.assertEqual(res_frame.headers["content-type"], "image/jpeg")
        self.assertGreater(len(res_frame.content), 0)

        # 2. Test video info
        res_info = self.client.get("/api/video/info")
        self.assertEqual(res_info.status_code, 200)
        info = res_info.json()
        self.assertIn("n_frames", info)

        # 3. Test video source streaming
        res_vid = self.client.get("/api/video/source")
        self.assertEqual(res_vid.status_code, 200)
        self.assertEqual(res_vid.headers["content-type"], "video/mp4")

    def test_export_and_download_endpoints(self):
        # Trigger export
        res = self.client.post("/api/export", json={
            "format": "stl",
            "level": 0.5,
            "as_negative_mold": False,
            "as_hollow": False,
            "step_size": 1,
            "smooth_iterations": 0,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        filename = data["filename"]

        # Test download
        dl_res = self.client.get(f"/api/download/{filename}")
        self.assertEqual(dl_res.status_code, 200)
        self.assertGreater(len(dl_res.content), 0)

    def test_export_with_constant_boundary(self):
        # Trigger export with rounded_rect framing
        res = self.client.post("/api/export", json={
            "format": "stl",
            "level": 0.5,
            "constant_boundary": "rounded_rect",
            "boundary_mode": "framed",
            "boundary_bezel": 3.0,
            "base_plate_slices": 2,
            "step_size": 1,
            "smooth_iterations": 0,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["stats"]["is_watertight"])
        self.assertGreater(data["stats"]["volume_mm3"], 0)


    def test_upload_video_endpoint(self):
        import tempfile
        from pathlib import Path
        from engine.dummy_data import write_dummy_mp4

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_clip = Path(tmp_dir) / "clip.mp4"
            write_dummy_mp4(test_clip, n=12)

            with open(test_clip, "rb") as f:
                res = self.client.post(
                    "/api/upload/video",
                    files={"file": ("clip.mp4", f, "video/mp4")},
                    data={
                        "use_flow": "true",
                        "interp_factor": "2",
                        "max_frames": "16",
                        "max_dimension": "128",
                        "start_frame": "0",
                        "end_frame": "12",
                    },
                )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "success")
            self.assertIn("shape", data)
            self.assertIn("duration_s", data)
            self.assertIn("probe", data)
            self.assertTrue(data["probe"]["valid"])

        # Test video probe endpoint
        res_probe = self.client.get("/api/video/probe")
        self.assertEqual(res_probe.status_code, 200)
        probe_data = res_probe.json()
        self.assertTrue(probe_data.get("exists", False))

        # Test video trim endpoint
        res_trim = self.client.post("/api/video/trim", json={
            "start_frame": 2,
            "end_frame": 8,
            "max_frames": 0,
            "max_dimension": 128,
            "use_flow": False,
        })
        self.assertEqual(res_trim.status_code, 200)
        trim_data = res_trim.json()
        self.assertEqual(trim_data["status"], "success")
        self.assertEqual(trim_data["shape"][0], 6)  # 8 - 2 = 6 frames


if __name__ == "__main__":
    unittest.main()

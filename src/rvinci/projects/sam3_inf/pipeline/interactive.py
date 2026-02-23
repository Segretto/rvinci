import cv2

class InteractiveLabeler:
    """Tool to interactively pick points on a video frame."""
    def __init__(self, image_path):
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Could not load {image_path}")
        self.display = self.image.copy()
        self.points = []
        self.current_obj_id = 1
        self.finished = False
        self.window_name = "Interactive Labeling"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append({"obj_id": self.current_obj_id, "point": [x, y], "label": 1})
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append({"obj_id": self.current_obj_id, "point": [x, y], "label": 0})
        self.update_display()

    def update_display(self):
        self.display = self.image.copy()

        for p in self.points:
            color = (0, 255, 0) if p["label"] == 1 else (0, 0, 255)
            cv2.circle(self.display, tuple(p["point"]), 5, color, -1)
            cv2.putText(self.display, str(p["obj_id"]),
                        (p["point"][0] + 5, p["point"][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(self.display,
                    f"Obj: {self.current_obj_id} | n=next | 1-9=set | c=clear | q=finish",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

        cv2.imshow(self.window_name, self.display)

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.update_display()

        while not self.finished:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q") or key == 27:
                self.finished = True
            elif key == ord("n"):
                self.current_obj_id += 1
            elif key == ord("c"):
                self.points = []
            elif ord("1") <= key <= ord("9"):
                self.current_obj_id = int(chr(key))
            self.update_display()

        cv2.destroyAllWindows()
        return self.points


def format_inputs_for_sam3(user_inputs):
    grouped = {}
    for item in user_inputs:
        oid = item["obj_id"]
        if oid not in grouped:
            grouped[oid] = {"points": [], "labels": []}
        grouped[oid]["points"].append(item["point"])
        grouped[oid]["labels"].append(item["label"])

    obj_ids = sorted(grouped.keys())
    points_list = []
    labels_list = []

    for oid in obj_ids:
        points_list.append(grouped[oid]["points"])
        labels_list.append(grouped[oid]["labels"])

    return obj_ids, points_list, labels_list

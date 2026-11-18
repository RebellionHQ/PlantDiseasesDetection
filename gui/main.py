import asyncio
import base64
from datetime import datetime

import flet as ft

from classifier import confidence_color, is_healthy, load_image, predict, to_jpeg
from history import History

try:
    import flet_camera as fc
except ImportError:
    fc = None

GREEN = "#2F7D4F"
GREEN_SOFT = "#E8F3EB"
INK = "#182A1F"
MUTED = "#6C7D71"
LINE = "#E2EAE3"
WHITE = "#FFFFFF"
CANVAS = "#F1F6F0"
AMBER_SOFT = "#FFF2E0"

MAX_WIDTH = 460

# (name, number of diseases besides "healthy", has a "healthy" class)
SUPPORTED_PLANTS = [
    ("Apple", 3, True),
    ("Blueberry", 0, True),
    ("Cherry", 1, True),
    ("Corn", 3, True),
    ("Grape", 3, True),
    ("Orange", 1, False),
    ("Peach", 1, True),
    ("Pepper, bell", 1, True),
    ("Potato", 2, True),
    ("Raspberry", 0, True),
    ("Soybean", 0, True),
    ("Squash", 1, False),
    ("Strawberry", 1, True),
    ("Tomato", 9, True),
]

DISEASE_COUNT = sum(item[1] for item in SUPPORTED_PLANTS)


def image_src(data: bytes) -> str:
    """Turn image bytes into a data URI, which every Flet renderer can display."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def format_time(value: str) -> str:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return ""
    today = datetime.now().date()
    if moment.date() == today:
        prefix = "Today"
    elif (today - moment.date()).days == 1:
        prefix = "Yesterday"
    else:
        prefix = moment.strftime("%d %b")
    return f"{prefix}, {moment.strftime('%H:%M')}"


def decode_image(entry: dict) -> bytes:
    try:
        return base64.b64decode(entry.get("image", ""))
    except ValueError:
        return b""


class PlantApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.history = History()
        self.picker = ft.FilePicker()
        self.scan = None
        self.analyzing = False
        self.tab = 0
        self.camera = None
        self.camera_ready = False
        self.camera_busy = False
        self.camera_error = None
        self.camera_shutter = None
        self.camera_status = None

        self.shell = ft.Container(width=MAX_WIDTH, expand=True, bgcolor=WHITE)
        self.root = ft.Column(
            controls=[self.shell],
            expand=True,
            spacing=0,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        page.title = "Plant Doctor"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.theme = ft.Theme(color_scheme_seed=GREEN)
        page.bgcolor = CANVAS
        page.padding = 0
        page.on_resize = self._on_resize

        self._paint()
        page.add(ft.SafeArea(content=self.root, expand=True))

    def _on_resize(self, _event=None):
        self._paint()
        self.page.update()

    def _render(self):
        self._paint()
        self.page.update()

    def _paint(self):
        width = self.page.width or MAX_WIDTH
        self.root.width = width
        self.shell.width = min(width, MAX_WIDTH)
        if self.camera is not None:
            self.shell.content = self._camera_layout()
        else:
            self.shell.content = ft.Column(
                controls=[self._header(), self._body(), self._nav()],
                spacing=0,
                expand=True,
            )

    def _toast(self, message: str):
        self.page.show_dialog(
            ft.SnackBar(
                content=ft.Text(message, color=WHITE, size=13),
                bgcolor=INK,
                behavior=ft.SnackBarBehavior.FLOATING,
                margin=ft.Margin.only(left=16, right=16, bottom=16),
            )
        )

    def _header(self):
        if self.tab == 1:
            count = len(self.history.entries)
            title = "History"
            subtitle = f"{count} scan" + ("" if count == 1 else "s")
            icon = ft.Icons.HISTORY
            trailing = (
                ft.IconButton(
                    ft.Icons.DELETE_OUTLINE,
                    icon_color=MUTED,
                    tooltip="Clear history",
                    on_click=self._confirm_clear,
                )
                if count
                else ft.Container(width=44)
            )
        elif self.tab == 2:
            title = "Plants & help"
            subtitle = f"{len(SUPPORTED_PLANTS)} plants · {DISEASE_COUNT} diseases"
            icon = ft.Icons.ECO
            trailing = ft.Container(width=44)
        else:
            title = "Plant Doctor"
            subtitle = "Scan a leaf to detect diseases"
            icon = ft.Icons.ECO
            trailing = ft.IconButton(
                ft.Icons.HELP_OUTLINE,
                icon_color=MUTED,
                tooltip="Which plants can I scan?",
                on_click=self._open_help,
            )

        return ft.Container(
            padding=ft.Padding.only(left=18, right=10, top=10, bottom=12),
            content=ft.Row(
                controls=[
                    ft.Container(
                        width=44,
                        height=44,
                        border_radius=14,
                        bgcolor=GREEN_SOFT,
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(icon, color=GREEN, size=23),
                    ),
                    ft.Column(
                        controls=[
                            ft.Text(
                                title,
                                size=21,
                                weight=ft.FontWeight.BOLD,
                                color=INK,
                            ),
                            ft.Text(subtitle, size=12, color=MUTED),
                        ],
                        spacing=0,
                        expand=True,
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    trailing,
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    def _body(self):
        if self.tab == 1:
            return self._history_body()
        if self.tab == 2:
            return self._help_body()
        return ft.Container(
            expand=True,
            padding=ft.Padding.only(left=16, right=16, top=2),
            content=ft.Column(
                controls=[
                    self._preview(),
                    self._actions(),
                    self._result_area(),
                    ft.Row(
                        controls=[
                            ft.Text(
                                "Use a clear photo of a single leaf in daylight.",
                                size=11,
                                color=MUTED,
                                expand=True,
                            ),
                            ft.TextButton(
                                "Which plants?",
                                icon=ft.Icons.HELP_OUTLINE,
                                on_click=self._open_help,
                                style=ft.ButtonStyle(
                                    padding=ft.Padding.symmetric(
                                        horizontal=6, vertical=0
                                    ),
                                    text_style=ft.TextStyle(size=11),
                                    color=GREEN,
                                    icon_color=GREEN,
                                ),
                            ),
                        ],
                        spacing=6,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=14,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
        )

    def _nav(self):
        return ft.Container(
            border=ft.Border.only(top=ft.BorderSide(1, LINE)),
            content=ft.NavigationBar(
                selected_index=self.tab,
                on_change=self._select_tab,
                bgcolor=WHITE,
                indicator_color=GREEN_SOFT,
                label_behavior=ft.NavigationBarLabelBehavior.ALWAYS_SHOW,
                destinations=[
                    ft.NavigationBarDestination(
                        icon=ft.Icons.CAMERA_ALT_OUTLINED,
                        selected_icon=ft.Icons.CAMERA_ALT,
                        label="Scan",
                    ),
                    ft.NavigationBarDestination(
                        icon=ft.Icons.HISTORY_OUTLINED,
                        selected_icon=ft.Icons.HISTORY,
                        label="History",
                    ),
                    ft.NavigationBarDestination(
                        icon=ft.Icons.ECO_OUTLINED,
                        selected_icon=ft.Icons.ECO,
                        label="Plants",
                    ),
                ],
            ),
        )

    def _select_tab(self, event):
        self.tab = event.control.selected_index
        self._render()

    def _open_help(self, _event=None):
        self.tab = 2
        self._render()

    def _preview(self):
        width = max(0, self.shell.width - 32)

        if self.scan is None:
            return ft.Container(
                width=width,
                height=248,
                border_radius=24,
                bgcolor="#F7FAF5",
                border=ft.Border.all(1, LINE),
                alignment=ft.Alignment.CENTER,
                content=ft.Column(
                    controls=[
                        ft.Container(
                            width=72,
                            height=72,
                            border_radius=36,
                            bgcolor=GREEN_SOFT,
                            alignment=ft.Alignment.CENTER,
                            content=ft.Icon(
                                ft.Icons.LOCAL_FLORIST, color=GREEN, size=34
                            ),
                        ),
                        ft.Text(
                            "No photo yet",
                            size=15,
                            weight=ft.FontWeight.W_600,
                            color=INK,
                        ),
                        ft.Text(
                            "Take a photo or pick one\nfrom your gallery",
                            size=12,
                            color=MUTED,
                            text_align=ft.TextAlign.CENTER,
                        ),
                    ],
                    spacing=10,
                    tight=True,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )

        image_data = self.scan.get("image") or b""
        image = ft.Container(
            width=width,
            height=248,
            border_radius=24,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            bgcolor="#DFE7DF",
            content=(
                ft.Image(
                    src=image_src(image_data),
                    fit=ft.BoxFit.COVER,
                    expand=True,
                    gapless_playback=True,
                    error_content=ft.Container(
                        alignment=ft.Alignment.CENTER,
                        content=ft.Column(
                            controls=[
                                ft.Icon(ft.Icons.BROKEN_IMAGE, color=MUTED, size=32),
                                ft.Text(
                                    "Could not display this image",
                                    size=12,
                                    color=MUTED,
                                ),
                            ],
                            spacing=8,
                            tight=True,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ),
                )
                if image_data
                else ft.Icon(ft.Icons.BROKEN_IMAGE, color=MUTED, size=32)
            ),
        )

        if not self.analyzing:
            return image

        return ft.Stack(
            width=width,
            height=248,
            fit=ft.StackFit.EXPAND,
            controls=[
                image,
                ft.Container(
                    border_radius=24,
                    bgcolor="#B3182A1F",
                    alignment=ft.Alignment.CENTER,
                    content=ft.Column(
                        controls=[
                            ft.ProgressRing(
                                width=32,
                                height=32,
                                stroke_width=3,
                                color=WHITE,
                            ),
                            ft.Text("Analyzing...", size=13, color=WHITE),
                        ],
                        spacing=12,
                        tight=True,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ),
            ],
        )

    def _actions(self):
        return ft.Row(
            controls=[
                ft.FilledButton(
                    content="Camera",
                    icon=ft.Icons.CAMERA_ALT,
                    on_click=self._open_camera,
                    disabled=self.analyzing,
                    expand=True,
                    height=48,
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=14),
                    ),
                ),
                ft.OutlinedButton(
                    content="Gallery",
                    icon=ft.Icons.PHOTO_LIBRARY,
                    on_click=self._pick_gallery,
                    disabled=self.analyzing,
                    expand=True,
                    height=48,
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=14),
                        side=ft.BorderSide(1, "#CBD9CE"),
                        color=INK,
                    ),
                ),
            ],
            spacing=12,
        )

    def _result_area(self):
        if self.scan is None:
            return ft.Container()
        if self.scan.get("error"):
            return ft.Container(
                padding=16,
                border_radius=18,
                bgcolor="#FBEDEC",
                content=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.ERROR_OUTLINE, color="#B23A2E", size=20),
                        ft.Text(
                            "Could not analyze this image. Try another photo.",
                            size=12,
                            color="#8E2F26",
                            expand=True,
                        ),
                    ],
                    spacing=10,
                ),
            )
        if self.scan.get("result") is None:
            return ft.Container()
        return self._result_card(self.scan["result"])

    def _result_card(self, result: dict):
        healthy = is_healthy(result["disease"])
        accent = GREEN if healthy else confidence_color(result["confidence"])
        badge_bg = GREEN_SOFT if healthy else AMBER_SOFT
        badge_color = GREEN if healthy else "#C77700"
        badge_text = "Healthy" if healthy else "Disease detected"
        confidence = result["confidence"]

        rows = []
        for index, item in enumerate(result.get("top") or []):
            rows.append(
                ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text(
                                    str(index + 1), size=12, color=MUTED, width=14
                                ),
                                ft.Text(
                                    f"{item['plant']} · {item['disease']}",
                                    size=12,
                                    color=INK,
                                    expand=True,
                                    max_lines=1,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                                ft.Text(
                                    f"{item['confidence']:.1f}%",
                                    size=12,
                                    color=MUTED,
                                ),
                            ],
                            spacing=8,
                        ),
                        ft.ProgressBar(
                            value=max(0.0, min(1.0, item["confidence"] / 100)),
                            color=accent if index == 0 else "#B9CDBE",
                            bgcolor="#EDF3EE",
                            bar_height=4,
                            border_radius=2,
                        ),
                    ],
                    spacing=5,
                )
            )

        return ft.Container(
            padding=18,
            bgcolor=WHITE,
            border_radius=22,
            border=ft.Border.all(1, LINE),
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text("Result", size=12, color=MUTED, expand=True),
                            ft.Container(
                                padding=ft.Padding.symmetric(
                                    horizontal=10, vertical=4
                                ),
                                bgcolor=badge_bg,
                                border_radius=999,
                                content=ft.Text(
                                    badge_text,
                                    size=11,
                                    color=badge_color,
                                    weight=ft.FontWeight.W_600,
                                ),
                            ),
                        ],
                    ),
                    ft.Column(
                        controls=[
                            ft.Text(
                                result["plant"],
                                size=22,
                                weight=ft.FontWeight.BOLD,
                                color=INK,
                            ),
                            ft.Text(result["disease"], size=15, color=INK),
                        ],
                        spacing=1,
                    ),
                    ft.Column(
                        controls=[
                            ft.Row(
                                controls=[
                                    ft.Text(
                                        "Confidence", size=12, color=MUTED, expand=True
                                    ),
                                    ft.Text(
                                        f"{confidence:.1f}%",
                                        size=13,
                                        weight=ft.FontWeight.W_600,
                                        color=accent,
                                    ),
                                ]
                            ),
                            ft.ProgressBar(
                                value=max(0.0, min(1.0, confidence / 100)),
                                color=accent,
                                bgcolor="#EDF3EE",
                                bar_height=8,
                                border_radius=4,
                            ),
                        ],
                        spacing=6,
                    ),
                    ft.Divider(height=1, color=LINE),
                    ft.Text("Top predictions", size=12, color=MUTED),
                    *rows,
                ],
                spacing=12,
            ),
        )

    def _history_body(self):
        entries = self.history.entries
        if not entries:
            return ft.Container(
                expand=True,
                padding=32,
                alignment=ft.Alignment.CENTER,
                content=ft.Column(
                    controls=[
                        ft.Icon(ft.Icons.HISTORY, color="#B9CDBE", size=52),
                        ft.Text(
                            "No scans yet",
                            size=16,
                            weight=ft.FontWeight.W_600,
                            color=INK,
                        ),
                        ft.Text(
                            "Your results will appear here.",
                            size=12,
                            color=MUTED,
                            text_align=ft.TextAlign.CENTER,
                        ),
                    ],
                    spacing=8,
                    tight=True,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )

        return ft.ListView(
            controls=[self._history_item(entry) for entry in entries],
            expand=True,
            spacing=10,
            padding=ft.Padding.only(left=16, right=16, top=2, bottom=16),
        )

    def _history_item(self, entry: dict):
        healthy = is_healthy(entry.get("disease", ""))
        accent = (
            GREEN
            if healthy
            else confidence_color(float(entry.get("confidence", 0)))
        )
        source_icon = (
            ft.Icons.CAMERA_ALT if entry.get("source") == "Camera" else ft.Icons.PHOTO_LIBRARY
        )
        image_data = decode_image(entry)

        thumbnail = ft.Container(
            width=52,
            height=52,
            border_radius=12,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            bgcolor="#EDF3EE",
            alignment=ft.Alignment.CENTER,
            content=ft.Image(src=image_src(image_data), fit=ft.BoxFit.COVER, expand=True)
            if image_data
            else ft.Icon(ft.Icons.IMAGE, color=MUTED, size=20),
        )

        return ft.Container(
            padding=10,
            border_radius=18,
            bgcolor=WHITE,
            border=ft.Border.all(1, LINE),
            on_click=lambda _event, item=entry: self._open_entry(item),
            content=ft.Row(
                controls=[
                    thumbnail,
                    ft.Column(
                        controls=[
                            ft.Text(
                                entry.get("plant", ""),
                                size=14,
                                weight=ft.FontWeight.W_600,
                                color=INK,
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                entry.get("disease", ""),
                                size=12,
                                color=MUTED,
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Row(
                                controls=[
                                    ft.Icon(source_icon, size=12, color=MUTED),
                                    ft.Text(
                                        format_time(entry.get("time", "")),
                                        size=10,
                                        color=MUTED,
                                    ),
                                ],
                                spacing=4,
                            ),
                        ],
                        spacing=1,
                        expand=True,
                    ),
                    ft.Text(
                        f"{float(entry.get('confidence', 0)):.0f}%",
                        size=13,
                        weight=ft.FontWeight.W_600,
                        color=accent,
                    ),
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    def _help_body(self):
        plant_rows = []
        for index in range(0, len(SUPPORTED_PLANTS), 2):
            pair = SUPPORTED_PLANTS[index : index + 2]
            cards = [self._plant_card(item) for item in pair]
            if len(cards) == 1:
                cards.append(ft.Container(expand=True))
            plant_rows.append(
                ft.Row(
                    controls=cards,
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                )
            )

        return ft.ListView(
            controls=[
                ft.Container(
                    padding=16,
                    bgcolor=GREEN_SOFT,
                    border_radius=20,
                    content=ft.Column(
                        controls=[
                            ft.Row(
                                controls=[
                                    ft.Icon(
                                        ft.Icons.HELP_OUTLINE, color=GREEN, size=20
                                    ),
                                    ft.Text(
                                        "What can I scan?",
                                        size=15,
                                        weight=ft.FontWeight.W_600,
                                        color=INK,
                                    ),
                                ],
                                spacing=8,
                            ),
                            ft.Text(
                                "The model recognizes "
                                f"{len(SUPPORTED_PLANTS)} plants and {DISEASE_COUNT} "
                                "leaf diseases, and can also tell when a leaf looks "
                                "healthy.",
                                size=12,
                                color=INK,
                            ),
                        ],
                        spacing=6,
                    ),
                ),
                ft.Text(
                    "Supported plants",
                    size=13,
                    weight=ft.FontWeight.W_600,
                    color=INK,
                ),
                *plant_rows,
                ft.Text(
                    "Photo tips",
                    size=13,
                    weight=ft.FontWeight.W_600,
                    color=INK,
                ),
                self._tips_card(),
                ft.Container(
                    padding=14,
                    bgcolor=AMBER_SOFT,
                    border_radius=18,
                    content=ft.Row(
                        controls=[
                            ft.Icon(
                                ft.Icons.INFO_OUTLINE, color="#C77700", size=20
                            ),
                            ft.Text(
                                "Only these plants are recognized — photos of other "
                                "plants may be misclassified. A photo without a leaf is "
                                "detected as \"Background\".\n\n"
                                "The camera works in a browser over localhost or HTTPS, "
                                "and in the mobile app. If it does not start, use "
                                "Gallery — it gives the same result.",
                                size=11,
                                color="#8A5A00",
                                expand=True,
                            ),
                        ],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                    ),
                ),
            ],
            expand=True,
            spacing=10,
            padding=ft.Padding.only(left=16, right=16, top=2, bottom=16),
        )

    def _plant_card(self, plant):
        name, diseases, healthy = plant
        if diseases == 0:
            subtitle = "Healthy check"
        else:
            subtitle = f"{diseases} disease" + ("" if diseases == 1 else "s")
            if healthy:
                subtitle += " + healthy"

        return ft.Container(
            expand=True,
            padding=10,
            border_radius=16,
            bgcolor=WHITE,
            border=ft.Border.all(1, LINE),
            content=ft.Row(
                controls=[
                    ft.Container(
                        width=36,
                        height=36,
                        border_radius=11,
                        bgcolor=GREEN_SOFT,
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(
                            ft.Icons.LOCAL_FLORIST, color=GREEN, size=18
                        ),
                    ),
                    ft.Column(
                        controls=[
                            ft.Text(
                                name,
                                size=13,
                                weight=ft.FontWeight.W_600,
                                color=INK,
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                subtitle,
                                size=10,
                                color=MUTED,
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=0,
                        expand=True,
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    def _tips_card(self):
        def tip(icon, title, text):
            return ft.Row(
                controls=[
                    ft.Container(
                        width=36,
                        height=36,
                        border_radius=11,
                        bgcolor="#F4F8F2",
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(icon, color=GREEN, size=18),
                    ),
                    ft.Column(
                        controls=[
                            ft.Text(
                                title,
                                size=12,
                                weight=ft.FontWeight.W_600,
                                color=INK,
                            ),
                            ft.Text(text, size=11, color=MUTED),
                        ],
                        spacing=0,
                        expand=True,
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

        return ft.Container(
            padding=14,
            border_radius=18,
            bgcolor=WHITE,
            border=ft.Border.all(1, LINE),
            content=ft.Column(
                controls=[
                    tip(
                        ft.Icons.CENTER_FOCUS_STRONG,
                        "One leaf per photo",
                        "Fill most of the frame with a single leaf.",
                    ),
                    tip(
                        ft.Icons.WB_SUNNY_OUTLINED,
                        "Good light",
                        "Daylight, no harsh shadows or flash glare.",
                    ),
                    tip(
                        ft.Icons.LANDSCAPE_OUTLINED,
                        "Plain background",
                        "A table, paper or the sky works best.",
                    ),
                    tip(
                        ft.Icons.HD_OUTLINED,
                        "Sharp and close",
                        "Avoid blurry, dark or very small leaves.",
                    ),
                ],
                spacing=12,
            ),
        )

    def _open_entry(self, entry: dict):
        self.scan = {
            "image": decode_image(entry),
            "result": {
                "plant": entry.get("plant", ""),
                "disease": entry.get("disease", ""),
                "confidence": float(entry.get("confidence", 0)),
                "top": entry.get("top") or [],
            },
            "error": None,
        }
        self.tab = 0
        self._render()

    def _confirm_clear(self, _event):
        def clear(_event):
            self.history.clear()
            self.page.pop_dialog()
            self._render()

        self.page.show_dialog(
            ft.AlertDialog(
                title=ft.Text("Clear history?"),
                content=ft.Text("All saved scans will be removed from this device."),
                actions=[
                    ft.TextButton(
                        "Cancel", on_click=lambda _event: self.page.pop_dialog()
                    ),
                    ft.FilledButton("Clear", on_click=clear),
                ],
            )
        )

    def _camera_supported(self) -> bool:
        if fc is None:
            return False
        return self.page.web or self.page.platform in (
            ft.PagePlatform.ANDROID,
            ft.PagePlatform.IOS,
        )

    def _insecure_web_hint(self) -> str:
        if not self.page.web:
            return ""
        url = (self.page.url or "").lower()
        if not url.startswith("http://"):
            return ""
        if url.startswith(("http://localhost", "http://127.0.0.1")):
            return ""
        return (
            "Browsers block the camera on a plain HTTP address. Open the app via "
            "http://localhost or over HTTPS, or use Gallery instead."
        )

    def _camera_error_hint(self, error: Exception) -> str:
        text = str(error).lower()
        if any(word in text for word in ("permission", "notallowed", "denied")):
            return (
                "Camera access is blocked. Click the camera icon in the browser "
                "address bar, allow access, then try again."
            )
        return self._insecure_web_hint()

    def _camera_dialog(self, message: str):
        self.page.show_dialog(
            ft.AlertDialog(
                title=ft.Text("Camera unavailable"),
                content=ft.Text(message, size=13),
                actions=[
                    ft.TextButton(
                        "Close", on_click=lambda _event: self.page.pop_dialog()
                    ),
                    ft.FilledButton(
                        "Use Gallery", on_click=self._dialog_use_gallery
                    ),
                ],
            )
        )

    async def _dialog_use_gallery(self, _event=None):
        await self._pick_gallery(None)
        self.page.pop_dialog()

    async def _camera_use_gallery(self, _event=None):
        self._close_camera()
        await self._pick_gallery(None)

    def _on_camera_state(self, event):
        if self.camera is None:
            return
        if getattr(event, "has_error", False):
            self.camera_error = (
                getattr(event, "error_description", None)
                or "The camera stopped unexpectedly."
            )
            self._render()

    async def _open_camera(self, _event):
        if fc is None:
            self._camera_dialog(
                "The flet-camera package is not installed. Install the "
                "dependencies from gui/requirements.txt to use the in-app "
                "camera, or pick a photo from the gallery."
            )
            return
        if not self._camera_supported():
            self._camera_dialog(
                "The live camera is only available in a web browser and in the "
                "Android/iOS app. In the desktop window, use Gallery — it gives "
                "the same result."
            )
            return
        if self.camera is not None:
            return

        try:
            self.camera = fc.Camera(
                expand=True,
                preview_enabled=True,
                on_state_change=self._on_camera_state,
                content=ft.Container(
                    alignment=ft.Alignment.CENTER,
                    content=ft.Icon(
                        ft.Icons.CENTER_FOCUS_STRONG, color="#FFFFFF66", size=44
                    ),
                ),
            )
            self.camera_ready = False
            self.camera_error = None
            self._render()
        except Exception as error:
            self.camera = None
            self.camera_ready = False
            self.camera_error = None
            self._render()
            self._camera_dialog(
                f"The camera could not be started: {error}\n\n"
                "You can still pick a photo from the gallery."
            )
            return

        try:
            cameras = await self.camera.get_available_cameras()
            if self.camera is None:
                return
            if not cameras:
                raise RuntimeError("No camera was found on this device.")
            back = next(
                (
                    item
                    for item in cameras
                    if item.lens_direction == fc.CameraLensDirection.BACK
                ),
                cameras[0],
            )
            await self.camera.initialize(
                description=back,
                resolution_preset=fc.ResolutionPreset.HIGH,
                enable_audio=False,
                image_format_group=fc.ImageFormatGroup.JPEG,
            )
            if self.camera is None:
                return
            if not self.page.web:
                await self.camera.lock_capture_orientation()
            if self.camera is None:
                return
            self.camera_ready = True
        except Exception as error:
            hint = self._camera_error_hint(error)
            self.camera_error = f"{error}" + (f"\n\n{hint}" if hint else "")
            self._render()
            return

        if self.camera_shutter is not None:
            self.camera_shutter.disabled = False
        if self.camera_status is not None:
            self.camera_status.value = "Frame a single leaf"
        self.page.update()

    def _close_camera(self, _event=None):
        self.camera = None
        self.camera_ready = False
        self.camera_busy = False
        self.camera_error = None
        self.camera_shutter = None
        self.camera_status = None
        self._render()

    def _camera_layout(self):
        header = ft.Container(
            padding=ft.Padding.only(left=8, right=8, top=8, bottom=8),
            content=ft.Row(
                controls=[
                    ft.IconButton(
                        ft.Icons.CLOSE,
                        icon_color=INK,
                        on_click=self._close_camera,
                    ),
                    ft.Text(
                        "Camera",
                        size=16,
                        weight=ft.FontWeight.W_600,
                        color=INK,
                        expand=True,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Container(width=48),
                ],
                spacing=4,
            ),
        )

        if self.camera_error:
            body = ft.Container(
                expand=True,
                padding=28,
                alignment=ft.Alignment.CENTER,
                content=ft.Column(
                    controls=[
                        ft.Icon(ft.Icons.ERROR_OUTLINE, color=MUTED, size=48),
                        ft.Text(
                            "Camera unavailable",
                            size=16,
                            weight=ft.FontWeight.W_600,
                            color=INK,
                        ),
                        ft.Text(
                            self.camera_error,
                            size=12,
                            color=MUTED,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Row(
                            controls=[
                                ft.OutlinedButton(
                                    "Close",
                                    on_click=self._close_camera,
                                    style=ft.ButtonStyle(
                                        shape=ft.RoundedRectangleBorder(radius=12)
                                    ),
                                ),
                                ft.FilledButton(
                                    "Use Gallery",
                                    icon=ft.Icons.PHOTO_LIBRARY,
                                    on_click=self._camera_use_gallery,
                                    style=ft.ButtonStyle(
                                        shape=ft.RoundedRectangleBorder(radius=12)
                                    ),
                                ),
                            ],
                            spacing=10,
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                    ],
                    spacing=12,
                    tight=True,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )
            return ft.Column(controls=[header, body], spacing=0, expand=True)

        preview = ft.Container(
            expand=True,
            margin=ft.Margin.symmetric(horizontal=14),
            border_radius=26,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            bgcolor="#000000",
            content=self.camera,
        )

        self.camera_shutter = ft.FilledIconButton(
            icon=ft.Icons.PHOTO_CAMERA,
            icon_size=32,
            width=76,
            height=76,
            style=ft.ButtonStyle(shape=ft.CircleBorder()),
            on_click=self._capture,
            disabled=not self.camera_ready,
        )
        self.camera_status = ft.Text(
            "Frame a single leaf" if self.camera_ready else "Starting camera...",
            size=12,
            color=MUTED,
        )

        footer = ft.Container(
            padding=ft.Padding.only(top=20, bottom=28),
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                controls=[self.camera_shutter, self.camera_status],
                spacing=12,
                tight=True,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        return ft.Column(
            controls=[header, preview, footer], spacing=0, expand=True
        )

    async def _capture(self, _event):
        if not self.camera_ready or self.camera is None or self.camera_busy:
            return
        self.camera_busy = True
        try:
            data = await self.camera.take_picture()
        except Exception as error:
            self._toast(f"Could not take a photo: {error}")
            self.camera_busy = False
            return
        self.camera_busy = False
        self._close_camera()
        await self._analyze(data, "Camera")

    async def _pick_gallery(self, _event):
        try:
            files = await self.picker.pick_files(
                file_type=ft.FilePickerFileType.IMAGE,
                allow_multiple=False,
                with_data=True,
                compression_quality=90,
            )
        except Exception as error:
            self._toast(f"Could not open the gallery: {error}")
            return
        if not files:
            return

        file = files[0]
        data = file.bytes
        if not data and file.path:
            try:
                with open(file.path, "rb") as source:
                    data = source.read()
            except OSError:
                data = None
        if not data:
            self._toast("Could not read the selected image.")
            return

        await self._analyze(data, "Gallery")

    async def _analyze(self, data: bytes, source: str):
        try:
            image = load_image(data)
        except Exception:
            self._toast("That file is not a supported image.")
            return

        self.tab = 0
        self.scan = {
            "image": to_jpeg(image, 900, 85),
            "result": None,
            "error": None,
        }
        self.analyzing = True
        self._render()

        try:
            result = await asyncio.to_thread(predict, image)
        except Exception as error:
            self.analyzing = False
            self.scan["error"] = str(error)
            self._render()
            return

        self.analyzing = False
        self.scan["result"] = result
        self.history.add(result, to_jpeg(image, 480, 80), source)
        self._render()


def main(page: ft.Page):
    PlantApp(page)


if __name__ == "__main__":
    # Serve the app on localhost and open it in the browser. The live camera
    # is not available in Flet's native desktop window, but works on localhost.
    ft.run(
        main,
        view=ft.AppView.WEB_BROWSER,
        host="127.0.0.1",
        port=8550,
    )

# auto created module
from pyfemtet.opt.visualization._wrapped_components.str_enum import StrEnum
# from enum import StrEnum
import dash
import dash_bootstrap_components


class Checklist(dash.dcc.Checklist):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        options = "options"
        value = "value"
        inline = "inline"
        className = "className"
        style = "style"
        inputStyle = "inputStyle"
        inputClassName = "inputClassName"
        labelStyle = "labelStyle"
        labelClassName = "labelClassName"
        id = "id"
        loading_state = "loading_state"
        persistence = "persistence"
        persisted_props = "persisted_props"
        persistence_type = "persistence_type"


class Clipboard(dash.dcc.Clipboard):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        id = "id"
        target_id = "target_id"
        content = "content"
        n_clicks = "n_clicks"
        html_content = "html_content"
        title = "title"
        style = "style"
        className = "className"
        loading_state = "loading_state"


class ConfirmDialog(dash.dcc.ConfirmDialog):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        id = "id"
        message = "message"
        submit_n_clicks = "submit_n_clicks"
        submit_n_clicks_timestamp = "submit_n_clicks_timestamp"
        cancel_n_clicks = "cancel_n_clicks"
        cancel_n_clicks_timestamp = "cancel_n_clicks_timestamp"
        displayed = "displayed"


class ConfirmDialogProvider(dash.dcc.ConfirmDialogProvider):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        children = "children"
        id = "id"
        message = "message"
        submit_n_clicks = "submit_n_clicks"
        submit_n_clicks_timestamp = "submit_n_clicks_timestamp"
        cancel_n_clicks = "cancel_n_clicks"
        cancel_n_clicks_timestamp = "cancel_n_clicks_timestamp"
        displayed = "displayed"
        loading_state = "loading_state"


class DatePickerRange(dash.dcc.DatePickerRange):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        start_date = "start_date"
        end_date = "end_date"
        min_date_allowed = "min_date_allowed"
        max_date_allowed = "max_date_allowed"
        disabled_days = "disabled_days"
        minimum_nights = "minimum_nights"
        updatemode = "updatemode"
        start_date_placeholder_text = "start_date_placeholder_text"
        end_date_placeholder_text = "end_date_placeholder_text"
        initial_visible_month = "initial_visible_month"
        clearable = "clearable"
        reopen_calendar_on_clear = "reopen_calendar_on_clear"
        display_format = "display_format"
        month_format = "month_format"
        first_day_of_week = "first_day_of_week"
        show_outside_days = "show_outside_days"
        stay_open_on_select = "stay_open_on_select"
        calendar_orientation = "calendar_orientation"
        number_of_months_shown = "number_of_months_shown"
        with_portal = "with_portal"
        with_full_screen_portal = "with_full_screen_portal"
        day_size = "day_size"
        is_RTL = "is_RTL"
        disabled = "disabled"
        start_date_id = "start_date_id"
        end_date_id = "end_date_id"
        style = "style"
        className = "className"
        id = "id"
        loading_state = "loading_state"
        persistence = "persistence"
        persisted_props = "persisted_props"
        persistence_type = "persistence_type"


class DatePickerSingle(dash.dcc.DatePickerSingle):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        date = "date"
        min_date_allowed = "min_date_allowed"
        max_date_allowed = "max_date_allowed"
        disabled_days = "disabled_days"
        placeholder = "placeholder"
        initial_visible_month = "initial_visible_month"
        clearable = "clearable"
        reopen_calendar_on_clear = "reopen_calendar_on_clear"
        display_format = "display_format"
        month_format = "month_format"
        first_day_of_week = "first_day_of_week"
        show_outside_days = "show_outside_days"
        stay_open_on_select = "stay_open_on_select"
        calendar_orientation = "calendar_orientation"
        number_of_months_shown = "number_of_months_shown"
        with_portal = "with_portal"
        with_full_screen_portal = "with_full_screen_portal"
        day_size = "day_size"
        is_RTL = "is_RTL"
        disabled = "disabled"
        style = "style"
        className = "className"
        id = "id"
        loading_state = "loading_state"
        persistence = "persistence"
        persisted_props = "persisted_props"
        persistence_type = "persistence_type"


class Download(dash.dcc.Download):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        id = "id"
        data = "data"
        base64 = "base64"
        type = "type"


class Dropdown(dash.dcc.Dropdown):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        options = "options"
        value = "value"
        multi = "multi"
        clearable = "clearable"
        searchable = "searchable"
        search_value = "search_value"
        placeholder = "placeholder"
        disabled = "disabled"
        optionHeight = "optionHeight"
        maxHeight = "maxHeight"
        style = "style"
        className = "className"
        id = "id"
        loading_state = "loading_state"
        persistence = "persistence"
        persisted_props = "persisted_props"
        persistence_type = "persistence_type"


class Geolocation(dash.dcc.Geolocation):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        id = "id"
        local_date = "local_date"
        timestamp = "timestamp"
        position = "position"
        position_error = "position_error"
        show_alert = "show_alert"
        update_now = "update_now"
        high_accuracy = "high_accuracy"
        maximum_age = "maximum_age"
        timeout = "timeout"


class Graph(dash.dcc.Graph):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        id = "id"
        responsive = "responsive"
        clickData = "clickData"
        clickAnnotationData = "clickAnnotationData"
        hoverData = "hoverData"
        clear_on_unhover = "clear_on_unhover"
        selectedData = "selectedData"
        relayoutData = "relayoutData"
        extendData = "extendData"
        prependData = "prependData"
        restyleData = "restyleData"
        figure = "figure"
        style = "style"
        className = "className"
        mathjax = "mathjax"
        animate = "animate"
        animation_options = "animation_options"
        config = "config"
        loading_state = "loading_state"


class Input(dash.dcc.Input):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        value = "value"
        type = "type"
        debounce = "debounce"
        placeholder = "placeholder"
        n_submit = "n_submit"
        n_submit_timestamp = "n_submit_timestamp"
        inputMode = "inputMode"
        autoComplete = "autoComplete"
        readOnly = "readOnly"
        required = "required"
        autoFocus = "autoFocus"
        disabled = "disabled"
        list = "list"
        multiple = "multiple"
        spellCheck = "spellCheck"
        name = "name"
        min = "min"
        max = "max"
        step = "step"
        minLength = "minLength"
        maxLength = "maxLength"
        pattern = "pattern"
        selectionStart = "selectionStart"
        selectionEnd = "selectionEnd"
        selectionDirection = "selectionDirection"
        n_blur = "n_blur"
        n_blur_timestamp = "n_blur_timestamp"
        size = "size"
        style = "style"
        className = "className"
        id = "id"
        loading_state = "loading_state"
        persistence = "persistence"
        persisted_props = "persisted_props"
        persistence_type = "persistence_type"


class Interval(dash.dcc.Interval):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        id = "id"
        interval = "interval"
        disabled = "disabled"
        n_intervals = "n_intervals"
        max_intervals = "max_intervals"


class Link(dash.dcc.Link):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        children = "children"
        href = "href"
        target = "target"
        refresh = "refresh"
        title = "title"
        className = "className"
        style = "style"
        id = "id"
        loading_state = "loading_state"


class Loading(dash.dcc.Loading):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        children = "children"
        id = "id"
        type = "type"
        fullscreen = "fullscreen"
        debug = "debug"
        className = "className"
        parent_className = "parent_className"
        style = "style"
        parent_style = "parent_style"
        overlay_style = "overlay_style"
        color = "color"
        loading_state = "loading_state"
        display = "display"
        delay_hide = "delay_hide"
        delay_show = "delay_show"
        show_initially = "show_initially"
        target_components = "target_components"
        custom_spinner = "custom_spinner"


class Location(dash.dcc.Location):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        id = "id"
        pathname = "pathname"
        search = "search"
        hash = "hash"
        href = "href"
        refresh = "refresh"


class LogoutButton(dash.dcc.LogoutButton):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        id = "id"
        label = "label"
        logout_url = "logout_url"
        style = "style"
        method = "method"
        className = "className"
        loading_state = "loading_state"


class Markdown(dash.dcc.Markdown):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        children = "children"
        id = "id"
        className = "className"
        mathjax = "mathjax"
        dangerously_allow_html = "dangerously_allow_html"
        link_target = "link_target"
        dedent = "dedent"
        highlight_config = "highlight_config"
        loading_state = "loading_state"
        style = "style"


class RadioItems(dash.dcc.RadioItems):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        options = "options"
        value = "value"
        inline = "inline"
        style = "style"
        className = "className"
        inputStyle = "inputStyle"
        inputClassName = "inputClassName"
        labelStyle = "labelStyle"
        labelClassName = "labelClassName"
        id = "id"
        loading_state = "loading_state"
        persistence = "persistence"
        persisted_props = "persisted_props"
        persistence_type = "persistence_type"


class RangeSlider(dash.dcc.RangeSlider):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        min = "min"
        max = "max"
        step = "step"
        marks = "marks"
        value = "value"
        drag_value = "drag_value"
        allowCross = "allowCross"
        pushable = "pushable"
        disabled = "disabled"
        count = "count"
        dots = "dots"
        included = "included"
        tooltip = "tooltip"
        updatemode = "updatemode"
        vertical = "vertical"
        verticalHeight = "verticalHeight"
        className = "className"
        id = "id"
        loading_state = "loading_state"
        persistence = "persistence"
        persisted_props = "persisted_props"
        persistence_type = "persistence_type"


class Slider(dash.dcc.Slider):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        min = "min"
        max = "max"
        step = "step"
        marks = "marks"
        value = "value"
        drag_value = "drag_value"
        disabled = "disabled"
        dots = "dots"
        included = "included"
        tooltip = "tooltip"
        updatemode = "updatemode"
        vertical = "vertical"
        verticalHeight = "verticalHeight"
        className = "className"
        id = "id"
        loading_state = "loading_state"
        persistence = "persistence"
        persisted_props = "persisted_props"
        persistence_type = "persistence_type"


class Store(dash.dcc.Store):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        id = "id"
        storage_type = "storage_type"
        data = "data"
        clear_data = "clear_data"
        modified_timestamp = "modified_timestamp"


class Tab(dash.dcc.Tab):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        children = "children"
        id = "id"
        label = "label"
        value = "value"
        disabled = "disabled"
        disabled_style = "disabled_style"
        disabled_className = "disabled_className"
        className = "className"
        selected_className = "selected_className"
        style = "style"
        selected_style = "selected_style"
        loading_state = "loading_state"


class Tabs(dash.dcc.Tabs):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        children = "children"
        id = "id"
        value = "value"
        className = "className"
        content_className = "content_className"
        parent_className = "parent_className"
        style = "style"
        parent_style = "parent_style"
        content_style = "content_style"
        vertical = "vertical"
        mobile_breakpoint = "mobile_breakpoint"
        colors = "colors"
        loading_state = "loading_state"
        persistence = "persistence"
        persisted_props = "persisted_props"
        persistence_type = "persistence_type"


class Textarea(dash.dcc.Textarea):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        id = "id"
        value = "value"
        autoFocus = "autoFocus"
        cols = "cols"
        disabled = "disabled"
        form = "form"
        maxLength = "maxLength"
        minLength = "minLength"
        name = "name"
        placeholder = "placeholder"
        readOnly = "readOnly"
        required = "required"
        rows = "rows"
        wrap = "wrap"
        accessKey = "accessKey"
        className = "className"
        contentEditable = "contentEditable"
        contextMenu = "contextMenu"
        dir = "dir"
        draggable = "draggable"
        hidden = "hidden"
        lang = "lang"
        spellCheck = "spellCheck"
        style = "style"
        tabIndex = "tabIndex"
        title = "title"
        n_blur = "n_blur"
        n_blur_timestamp = "n_blur_timestamp"
        n_clicks = "n_clicks"
        n_clicks_timestamp = "n_clicks_timestamp"
        loading_state = "loading_state"
        persistence = "persistence"
        persisted_props = "persisted_props"
        persistence_type = "persistence_type"


class Tooltip(dash.dcc.Tooltip):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        children = "children"
        id = "id"
        className = "className"
        style = "style"
        bbox = "bbox"
        show = "show"
        direction = "direction"
        border_color = "border_color"
        background_color = "background_color"
        loading_text = "loading_text"
        zindex = "zindex"
        targetable = "targetable"
        loading_state = "loading_state"


class Upload(dash.dcc.Upload):
    def _dummy(self):
        # noinspection PyAttributeOutsideInit
        self.id = None

    class Prop(StrEnum):
        children = "children"
        id = "id"
        contents = "contents"
        filename = "filename"
        last_modified = "last_modified"
        accept = "accept"
        disabled = "disabled"
        disable_click = "disable_click"
        max_size = "max_size"
        min_size = "min_size"
        multiple = "multiple"
        className = "className"
        className_active = "className_active"
        className_reject = "className_reject"
        className_disabled = "className_disabled"
        style = "style"
        style_active = "style_active"
        style_reject = "style_reject"
        style_disabled = "style_disabled"
        loading_state = "loading_state"



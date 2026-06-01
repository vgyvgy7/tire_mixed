from __future__ import annotations

import itertools
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from loading_engine import (
    CONTAINER_PRESETS,
    PALETTE,
    Container,
    calculate_mixed_loading,
    build_single_result_table,
    create_excel_report,
    override_tire_dimensions,
    parse_tire_size,
    validate_placements,
    write_logic_report,
)


N_SEG = 28


def parse_specs(multiline_text: str):
    specs = [s.strip() for s in multiline_text.replace(",", "\n").splitlines() if s.strip()]
    tires = []
    errors = []
    for idx, spec in enumerate(specs):
        try:
            tires.append(parse_tire_size(spec, PALETTE[idx % len(PALETTE)]))
        except ValueError as exc:
            errors.append(str(exc))
    return tires, errors


def container_wireframe(container: Container, offset_x: float = 0.0, name: str = "Container"):
    corners = np.array(list(itertools.product([0, container.length_mm], [0, container.width_mm], [0, container.height_mm])))
    corners[:, 0] += offset_x
    xw, yw, zw = [], [], []
    for s, e in combinations(corners, 2):
        if np.sum(np.abs(s - e)) in [container.length_mm, container.width_mm, container.height_mm]:
            xw += [s[0], e[0], None]
            yw += [s[1], e[1], None]
            zw += [s[2], e[2], None]
    return go.Scatter3d(
        x=xw,
        y=yw,
        z=zw,
        mode="lines",
        line=dict(color="rgba(15,23,42,0.28)", width=2),
        name=name,
        showlegend=False,
        hoverinfo="none",
    )


def cylinder_batch(items):
    theta = np.linspace(0, 2 * np.pi, N_SEG, endpoint=False)
    ct, st_ = np.cos(theta), np.sin(theta)
    ax, ay, az, ai, aj, ak = [], [], [], [], [], []
    off = 0
    for p in items:
        radius = float(p["outer_diameter_mm"]) / 2
        sw = float(p["section_width_mm"])
        cx, cy, cz = float(p["x"]), float(p["y"]), float(p["z"])
        axis = str(p.get("axis", "z"))
        if axis == "x":
            xb = np.full(N_SEG, cx)
            yb = radius * ct + cy
            zb = radius * st_ + cz
            xt = np.full(N_SEG, cx + sw)
            yt = radius * ct + cy
            zt = radius * st_ + cz
            c0 = (cx, cy, cz)
            c1 = (cx + sw, cy, cz)
        elif axis == "y":
            xb = radius * ct + cx
            yb = np.full(N_SEG, cy)
            zb = radius * st_ + cz
            xt = radius * ct + cx
            yt = np.full(N_SEG, cy + sw)
            zt = radius * st_ + cz
            c0 = (cx, cy, cz)
            c1 = (cx, cy + sw, cz)
        else:
            xb = radius * ct + cx
            yb = radius * st_ + cy
            zb = np.full(N_SEG, cz)
            xt = radius * ct + cx
            yt = radius * st_ + cy
            zt = np.full(N_SEG, cz + sw)
            c0 = (cx, cy, cz)
            c1 = (cx, cy, cz + sw)
        xs = np.concatenate([xb, xt, [c0[0], c1[0]]])
        ys = np.concatenate([yb, yt, [c0[1], c1[1]]])
        zs = np.concatenate([zb, zt, [c0[2], c1[2]]])
        for v in range(N_SEG):
            vn = (v + 1) % N_SEG
            ai += [off + v, off + vn]
            aj += [off + vn, off + v + N_SEG]
            ak += [off + v + N_SEG, off + vn + N_SEG]
        for v in range(N_SEG):
            ai.append(off + 2 * N_SEG)
            aj.append(off + v)
            ak.append(off + (v + 1) % N_SEG)
        for v in range(N_SEG):
            ai.append(off + 2 * N_SEG + 1)
            aj.append(off + v + N_SEG)
            ak.append(off + (v + 1) % N_SEG + N_SEG)
        ax.extend(xs)
        ay.extend(ys)
        az.extend(zs)
        off += 2 * N_SEG + 2
    return np.array(ax), np.array(ay), np.array(az), ai, aj, ak


def build_3d(container: Container, placements, title: str, container_count: int = 1):
    fig = go.Figure()
    gap = max(container.length_mm * 0.08, 800)
    for idx in range(container_count):
        fig.add_trace(container_wireframe(container, idx * (container.length_mm + gap), f"Container {idx + 1}"))
    df = pd.DataFrame(placements)
    if not df.empty:
        for spec, group in df.groupby("tire_spec", sort=False):
            items = group.to_dict("records")
            x, y, z, i, j, k = cylinder_batch(items)
            color = str(group.iloc[0]["color"])
            fig.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    color=color,
                    opacity=0.84,
                    name=f"{spec} ({len(items):,})",
                    showlegend=True,
                    flatshading=False,
                    lighting=dict(ambient=0.45, diffuse=0.85, roughness=0.45, specular=0.25),
                    lightposition=dict(x=container.length_mm * 1.5, y=-container.width_mm, z=container.height_mm * 3),
                )
            )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b> | placement count: {len(placements):,}", font=dict(size=14, color="#0F172A")),
        scene=dict(
            xaxis=dict(title="Length X (mm)", range=[0, container.length_mm * container_count + gap * max(0, container_count - 1)], backgroundcolor="rgba(241,245,249,0.7)", gridcolor="#CBD5E1"),
            yaxis=dict(title="Width Y (mm)", range=[0, container.width_mm], backgroundcolor="rgba(248,250,252,0.7)", gridcolor="#CBD5E1"),
            zaxis=dict(title="Height Z (mm)", range=[0, container.height_mm], backgroundcolor="rgba(255,255,255,0.7)", gridcolor="#CBD5E1"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=42, b=0),
        height=620,
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.88)", bordercolor="#E2E8F0", borderwidth=1),
        paper_bgcolor="#FAFAFA",
    )
    return fig


def offset_visual_placements(container: Container, main_placements, overflow_placements):
    gap = max(container.length_mm * 0.08, 800)
    visual = [dict(p, container_no=int(p.get("container_no", 1))) for p in main_placements]
    for p in overflow_placements:
        item = dict(p)
        container_no = int(item.get("container_no", 2))
        offset = (container_no - 1) * (container.length_mm + gap)
        for key in ("x", "bbox_x0", "bbox_x1", "block_start_x", "block_end_x"):
            if key in item:
                item[key] = float(item[key]) + offset
        visual.append(item)
    container_count = max([int(p.get("container_no", 1)) for p in visual], default=1)
    return visual, container_count


def choose_segment(label, options, default=None, key=None):
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options, default=default or options[0], key=key)
    return st.radio(label, options, index=options.index(default or options[0]), horizontal=True, key=key)


def input_dataframe(container, clearance, method, tires, allocation_mode, allocations, layout_axis):
    rows = [
        {"item": "container_L_mm", "value": container.length_mm},
        {"item": "container_W_mm", "value": container.width_mm},
        {"item": "container_H_mm", "value": container.height_mm},
        {"item": "clearance_mm", "value": clearance},
        {"item": "selected_loading_method", "value": method},
        {"item": "mixed_allocation_mode", "value": allocation_mode},
        {"item": "mixed_layout_axis", "value": layout_axis},
    ]
    for idx, tire in enumerate(tires):
        rows.append({"item": f"tire_{idx + 1}_spec", "value": tire.spec})
        rows.append({"item": f"tire_{idx + 1}_outer_diameter_mm", "value": round(tire.outer_diameter_mm, 1)})
        rows.append({"item": f"tire_{idx + 1}_section_width_mm", "value": round(tire.section_width_mm, 1)})
        rows.append({"item": f"tire_{idx + 1}_mixed_allocation", "value": allocations[idx] if idx < len(allocations) else ""})
    return pd.DataFrame(rows)


st.set_page_config(page_title="타이어 적재 시뮬레이션", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
html, body, [class*="css"], .stApp { font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif; background:#F8FAFC; }
#MainMenu, footer, header { visibility:hidden; }
[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"],
button[title="Close sidebar"], button[title="Open sidebar"],
button[aria-label="Close sidebar"], button[aria-label="Open sidebar"] { display:none !important; }
.block-container { max-width: 1440px; padding: 1.6rem 2rem 3rem; }
.app-header { display:flex; align-items:baseline; gap:1rem; border-bottom:2px solid #0F172A; padding-bottom:1rem; margin-bottom:1.2rem; }
.app-title { font-size:1.35rem; font-weight:750; color:#0F172A; }
.app-meta { color:#64748B; font-size:.78rem; font-family:Consolas, monospace; }
.small-note { color:#64748B; font-size:.82rem; }
</style>
<div class="app-header">
  <div class="app-title">타이어 평치 / 벌집 적재 시뮬레이션</div>
  <div class="app-meta">placement-driven loading calculator</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("컨테이너 / 차량")
    preset_name = st.selectbox("프리셋", list(CONTAINER_PRESETS.keys()), index=1)
    preset = CONTAINER_PRESETS[preset_name]
    length = st.number_input("컨테이너 길이 L (mm)", min_value=100.0, value=float(preset["length_mm"]), step=10.0)
    width = st.number_input("컨테이너 폭 W (mm)", min_value=100.0, value=float(preset["width_mm"]), step=10.0)
    height = st.number_input("컨테이너 높이 H (mm)", min_value=100.0, value=float(preset["height_mm"]), step=10.0)
    clearance = st.number_input("clearance / gap (mm)", min_value=0.0, value=0.0, step=1.0)
    method_label = st.radio("혼적 적재 방식", ["평치", "벌집"], horizontal=True)
    method = "flat" if method_label == "평치" else "honeycomb"

container = Container(length, width, height)

left, right = st.columns([0.48, 0.52], gap="large")

with left:
    st.subheader("타이어 규격 입력")
    spec_text = st.text_area(
        "여러 규격 입력",
        value="205/55R16\n195/65R15\n225/45R17\n200/30R14",
        height=128,
        help="쉼표 또는 줄바꿈으로 여러 규격을 입력할 수 있습니다.",
    )
    tires, errors = parse_specs(spec_text)
    for error in errors:
        st.error(error)

    st.caption("계산값 보정 옵션")
    adjusted_tires = []
    for idx, tire in enumerate(tires):
        with st.expander(f"{tire.spec} | 계산 OD {tire.outer_diameter_mm:.1f}mm / SW {tire.section_width_mm:.1f}mm", expanded=False):
            spec_key = tire.spec.replace("/", "_").replace(" ", "_")
            od = st.number_input(f"{tire.spec} outer_diameter_mm", min_value=1.0, value=float(round(tire.outer_diameter_mm, 1)), step=1.0, key=f"od_{idx}_{spec_key}")
            sw = st.number_input(f"{tire.spec} section_width_mm", min_value=1.0, value=float(round(tire.section_width_mm, 1)), step=1.0, key=f"sw_{idx}_{spec_key}")
            adjusted_tires.append(override_tire_dimensions(tire, od, sw))

    method_label = choose_segment("혼적 적재 방식", ["평치", "벌집"], default=method_label, key="mixed_method_tabs")
    method = "flat" if method_label == "평치" else "honeycomb"

    st.caption("혼적 배분 방식: 개수 지정")
    allocation_mode = "개수 지정"
    layout_axis = "x"
    layout_label = "세로 적재"

    default_alloc = [100.0, 100.0, 50.0, 0.0]
    suffix = "목표 개수"
    step = 1.0
    allocations = []
    for idx, tire in enumerate(adjusted_tires):
        default = default_alloc[idx] if idx < len(default_alloc) else 0.0
        label = f"{tire.spec} {suffix}"
        allocations.append(st.number_input(label, min_value=0.0, value=float(default), step=step, key=f"alloc_{idx}_{allocation_mode}_{layout_axis}"))

    run_single = st.button("단일 규격 계산", use_container_width=True)
    run_mixed = st.button("혼적 규격 계산", use_container_width=True)

positive_count_count = sum(1 for a in allocations if a > 0) if allocation_mode == "개수 지정" else 0
is_invalid_single_100_ratio = False
is_invalid_single_count_mix = allocation_mode == "개수 지정" and positive_count_count == 1
if is_invalid_single_count_mix:
    st.error("혼적이므로 다른 규격도 넣어야 합니다")

input_signature = (
    container.length_mm,
    container.width_mm,
    container.height_mm,
    clearance,
    method,
    allocation_mode,
    layout_axis,
    tuple((t.spec, round(t.outer_diameter_mm, 3), round(t.section_width_mm, 3)) for t in adjusted_tires),
    tuple(round(a, 3) for a in allocations),
)

if not adjusted_tires:
    st.session_state.single_df = pd.DataFrame()
    st.session_state.single_detail = {}
    st.session_state.mixed_result = {"placements": [], "table": pd.DataFrame(), "total_count": 0}
    st.session_state.input_signature = input_signature
elif is_invalid_single_100_ratio or is_invalid_single_count_mix:
    st.session_state.single_df, st.session_state.single_detail = build_single_result_table(container, adjusted_tires, clearance)
    st.session_state.mixed_result = {"placements": [], "table": pd.DataFrame(), "total_count": 0}
    st.session_state.input_signature = input_signature
elif st.session_state.get("input_signature") != input_signature:
    st.session_state.single_df, st.session_state.single_detail = build_single_result_table(container, adjusted_tires, clearance)
    st.session_state.mixed_result = calculate_mixed_loading(container, adjusted_tires, method, clearance, allocation_mode, allocations, layout_axis)
    st.session_state.input_signature = input_signature

if run_single and adjusted_tires:
    st.session_state.single_df, st.session_state.single_detail = build_single_result_table(container, adjusted_tires, clearance)
    st.session_state.input_signature = input_signature

if run_mixed and adjusted_tires and not is_invalid_single_100_ratio and not is_invalid_single_count_mix:
    st.session_state.mixed_result = calculate_mixed_loading(container, adjusted_tires, method, clearance, allocation_mode, allocations, layout_axis)
    st.session_state.input_signature = input_signature

single_df = st.session_state.get("single_df", pd.DataFrame())
single_detail = st.session_state.get("single_detail", {})
mixed_result = st.session_state.get("mixed_result", {"placements": [], "table": pd.DataFrame(), "total_count": 0})

with right:
    st.subheader("단일 규격별 최대 적입수량")
    if single_df.empty:
        st.info("타이어 규격을 입력하면 결과표가 생성됩니다.")
    else:
        st.dataframe(single_df, use_container_width=True, hide_index=True)

    st.subheader("혼적 결과")
    mixed_df = mixed_result["table"]
    if mixed_df.empty:
        st.info("혼적 배분값을 입력하면 결과표가 생성됩니다.")
    else:
        unplaced_total = int(mixed_result.get("unplaced_total") or 0)
        first_container_unplaced_total = int(mixed_result.get("first_container_unplaced_total") or 0)
        requested_total = mixed_result.get("requested_total")
        dedicated_loaded_total = int(mixed_result.get("dedicated_loaded_total") or 0)
        loaded_total = int(mixed_result.get("loaded_total") or mixed_result["total_count"])
        if dedicated_loaded_total > 0:
            st.info(f"단일 최대 적재량 기준으로 {dedicated_loaded_total:,}개는 단독 컨테이너로 분리하고, 잔량 {mixed_result['total_count']:,}개를 혼적 계산에 사용했습니다.")
            explanation_lines = [
                "혼적 개수 계산 방식",
                f"- 선택 방식: {'평치' if method == 'flat' else '벌집'}",
                "- 각 규격의 단일 최대 적재량을 먼저 계산합니다.",
                "- 입력 수량이 단일 최대 적재량보다 크면, 최대 적재량 단위는 단독 컨테이너로 분리합니다.",
                "- 단독 컨테이너에 들어가지 않고 남은 잔량만 현재 혼적 컨테이너에 배치합니다.",
                "- 혼적 잔량은 수량이 많은 규격부터 배치합니다.",
            ]
            for _, row in mixed_df.iterrows():
                original = int(row.get("original_requested_count") or 0)
                capacity = int(row.get("single_spec_capacity") or 0)
                dedicated_containers = int(row.get("dedicated_container_count") or 0)
                dedicated_count = int(row.get("dedicated_count") or 0)
                mixed_requested = int(row.get("mixed_requested_count") or 0)
                mixed_count = int(row.get("count") or 0)
                if original > 0:
                    explanation_lines.append(
                        f"- {row['tire_spec']}: 입력 {original:,}개 / 단일 최대 {capacity:,}개 / "
                        f"단독 {dedicated_containers:,}대({dedicated_count:,}개) / "
                        f"혼적 대상 {mixed_requested:,}개 / 실제 혼적 {mixed_count:,}개"
                    )
            st.markdown("\n".join(explanation_lines))
        if first_container_unplaced_total > 0:
            st.info(f"첫 번째 혼적 컨테이너에서 남은 {first_container_unplaced_total:,}개는 추가 컨테이너에 이어서 적재했습니다.")
        if unplaced_total > 0:
            st.warning(f"요청 {int(requested_total):,}개 중 총 {loaded_total:,}개 처리, {unplaced_total:,}개는 현재 컨테이너 구성에 미적재입니다.")
        st.dataframe(mixed_df, use_container_width=True, hide_index=True)
        overflow_df = mixed_result.get("overflow_table", pd.DataFrame())
        if isinstance(overflow_df, pd.DataFrame) and not overflow_df.empty:
            st.markdown("추가 컨테이너 적재")
            st.dataframe(overflow_df, use_container_width=True, hide_index=True)

tabs = st.tabs(["3D 시각화", "Validation", "Excel / 보고서"])

with tabs[0]:
    mode = st.radio("시각화 대상", ["혼적", "단일 규격 평치", "단일 규격 벌집"], horizontal=True)
    placements = []
    title = "혼적 적재"
    expected_count = None
    container_count = 1
    if mode == "혼적":
        placements, container_count = offset_visual_placements(
            container,
            mixed_result.get("placements", []),
            mixed_result.get("overflow_placements", []),
        )
        expected_count = len(placements)
        title = f"혼적 적재 | {method_label}"
    elif adjusted_tires:
        selected = st.selectbox("단일 규격 선택", [t.spec for t in adjusted_tires])
        if selected in single_detail:
            key = "flat" if mode == "단일 규격 평치" else "honeycomb"
            full_placements = single_detail[selected][key]["placements"]
            capacity = int(single_detail[selected][key]["count"])
            requested_single_count = int(st.number_input(
                "단일 규격 요청 수량",
                min_value=0,
                value=capacity,
                step=1,
                key=f"single_requested_{selected}_{key}",
            ))
            if requested_single_count <= capacity:
                visible_count = requested_single_count
                completed_containers = 0
            else:
                remainder = requested_single_count % capacity
                if remainder == 0:
                    visible_count = capacity
                    completed_containers = max(0, requested_single_count // capacity - 1)
                else:
                    visible_count = remainder
                    completed_containers = requested_single_count // capacity
            placements = full_placements[:visible_count]
            expected_count = visible_count
            method_text = "평치" if key == "flat" else "벌집"
            title = f"{selected} | {method_text} | 마지막 컨테이너"
            if requested_single_count > capacity:
                st.info(
                    f"{selected} {method_text} 단일 최대 적재량은 {capacity:,}개입니다. "
                    f"요청 {requested_single_count:,}개 중 {completed_containers:,}대는 최대 적재로 처리하고, "
                    f"마지막 컨테이너 {visible_count:,}개 적재 상태를 표시합니다."
                )
            else:
                st.info(
                    f"{selected} {method_text} 단일 최대 적재량은 {capacity:,}개입니다. "
                    f"요청 {requested_single_count:,}개 적재 상태를 표시합니다."
                )
    st.plotly_chart(build_3d(container, placements, title, container_count), use_container_width=True)
    st.markdown(f"<div class='small-note'>표시 타이어 수: {len(placements):,}개</div>", unsafe_allow_html=True)

with tabs[1]:
    validation_df = validate_placements(container, mixed_result["placements"], clearance, int(mixed_result["total_count"]))
    st.dataframe(validation_df, use_container_width=True, hide_index=True)
    if (validation_df["status"] == "FAIL").any():
        st.error("검증 실패 항목이 있습니다. clearance, 목표 개수, 컨테이너 크기를 확인하세요.")
    elif (validation_df["status"] == "WARN").any():
        st.warning("경고 항목이 있습니다. 필요하면 clearance를 조정하세요.")
    else:
        st.success("혼적 placement validation PASS")

with tabs[2]:
    input_df = input_dataframe(container, clearance, method, adjusted_tires, allocation_mode, allocations, layout_label)
    validation_df = validate_placements(container, mixed_result["placements"], clearance, int(mixed_result["total_count"]))
    excel_bytes = create_excel_report(input_df, single_df, mixed_df, validation_df)
    st.download_button(
        "Excel 다운로드",
        data=excel_bytes,
        file_name="tire_loading_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    if st.button("loading_logic_report.md 생성/갱신", use_container_width=True):
        write_logic_report(str(Path("loading_logic_report.md")))
        st.success("loading_logic_report.md 파일을 생성했습니다.")
    st.markdown("Excel 시트: INPUT, SINGLE_SPEC_RESULT, MIXED_LOADING_RESULT, LOGIC_EXPLANATION, VALIDATION")

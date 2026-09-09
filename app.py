
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Data-Driven Social Engagement Initiative",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #666;
        margin-bottom: 1.5rem;
    }
    .kpi {
        padding: 18px;
        border-radius: 12px;
        background: #f7f7f7;
        border: 1px solid #e6e6e6;
    }
    .section-note {
        padding: 12px 16px;
        border-radius: 10px;
        background: #f8f9fa;
        border-left: 4px solid #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FILES = [
    "Viral_Social_Media_Analytics_Ready.csv",
    "Cleaned_Viral_Social_Media_Trends.csv",
    "Cleaned_Viral_Social_Media_Trends_processed.csv",
    "viral_social_media_trends.csv",
]

REQUIRED_COLUMNS = [
    "Post_ID",
    "Post_Date",
    "Platform",
    "Hashtag",
    "Content_Type",
    "Region",
    "Views",
    "Likes",
    "Shares",
    "Comments",
    "Engagement_Level",
]


def find_default_csv():
    for filename in DEFAULT_FILES:
        path = BASE_DIR / filename
        if path.exists():
            return path
    return None


def normalise_columns(df):
    """Make common column-name variations compatible with the dashboard."""
    df = df.copy()
    original = {str(c).strip(): c for c in df.columns}

    aliases = {
        "Post ID": "Post_ID",
        "PostID": "Post_ID",
        "post_id": "Post_ID",
        "Post Date": "Post_Date",
        "Date": "Post_Date",
        "post_date": "Post_Date",
        "Platform": "Platform",
        "platform": "Platform",
        "Hashtag": "Hashtag",
        "hashtag": "Hashtag",
        "Content Type": "Content_Type",
        "ContentType": "Content_Type",
        "content_type": "Content_Type",
        "Region": "Region",
        "region": "Region",
        "Views": "Views",
        "views": "Views",
        "Likes": "Likes",
        "likes": "Likes",
        "Shares": "Shares",
        "shares": "Shares",
        "Comments": "Comments",
        "comments": "Comments",
        "Engagement Level": "Engagement_Level",
        "EngagementLevel": "Engagement_Level",
        "engagement_level": "Engagement_Level",
    }

    rename_map = {}
    for clean_name, original_name in original.items():
        if clean_name in aliases:
            rename_map[original_name] = aliases[clean_name]

    df = df.rename(columns=rename_map)

    # Case-insensitive fallback.
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for target in REQUIRED_COLUMNS:
        if target not in df.columns:
            candidates = [
                target.lower(),
                target.replace("_", " ").lower(),
            ]
            for candidate in candidates:
                if candidate in lower_map:
                    df = df.rename(columns={lower_map[candidate]: target})
                    break

    return df


def prepare_data(df):
    df = normalise_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ".\n\nExpected columns: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    df = df.copy()

    # Dates
    df["Post_Date"] = pd.to_datetime(df["Post_Date"], errors="coerce")

    # Numeric columns
    numeric_cols = ["Views", "Likes", "Shares", "Comments"]
    for col in numeric_cols:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Text columns
    text_cols = ["Post_ID", "Platform", "Hashtag", "Content_Type", "Region", "Engagement_Level"]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()

    # Clean invalid rows and duplicates
    df = df.drop_duplicates().copy()
    df = df.dropna(
        subset=["Post_ID", "Post_Date", "Views", "Likes", "Shares", "Comments"]
    ).copy()

    # Avoid divide-by-zero
    safe_views = df["Views"].replace(0, np.nan)

    # Module 1
    df["Engagement_Rate"] = (
        (df["Likes"] + df["Shares"] + df["Comments"]) / safe_views * 100
    )

    # Saves are not present in the supplied dataset.
    df["Save_to_Share_Ratio"] = np.nan

    # Module 2 - transparent viral coefficient.
    # Shares are weighted highest because Saves are unavailable.
    df["Viral_Coefficient"] = (
        0.60 * (df["Shares"] / safe_views)
        + 0.25 * (df["Comments"] / safe_views)
        + 0.15 * (df["Likes"] / safe_views)
    )

    df["Year_Month"] = df["Post_Date"].dt.to_period("M").astype(str)
    df["Month"] = df["Post_Date"].dt.month
    df["Year"] = df["Post_Date"].dt.year
    df["Day_of_Week"] = df["Post_Date"].dt.day_name()

    df["Engagement_Rate"] = df["Engagement_Rate"].replace([np.inf, -np.inf], np.nan)
    df["Viral_Coefficient"] = df["Viral_Coefficient"].replace(
        [np.inf, -np.inf], np.nan
    )

    return df


@st.cache_data(show_spinner=False)
def load_csv_from_path(path_string):
    raw = pd.read_csv(path_string)
    return prepare_data(raw)


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes):
    from io import BytesIO
    raw = pd.read_csv(BytesIO(file_bytes))
    return prepare_data(raw)


def fmt_number(value):
    if pd.isna(value):
        return "0"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.0f}"


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📊 Project Navigation")
st.sidebar.caption("Data-Driven Social Engagement Initiative")

uploaded_file = st.sidebar.file_uploader(
    "Upload social-media CSV",
    type=["csv"],
    help="Upload your Cleaned_Viral_Social_Media_Trends.csv or analytics-ready CSV.",
)

default_path = find_default_csv()

if uploaded_file is not None:
    try:
        df = load_csv_from_bytes(uploaded_file.getvalue())
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    except Exception as exc:
        st.error(f"Could not load the uploaded CSV: {exc}")
        st.stop()
elif default_path is not None:
    try:
        df = load_csv_from_path(str(default_path))
        st.sidebar.success(f"Loaded: {default_path.name}")
    except Exception as exc:
        st.error(f"Could not load {default_path.name}: {exc}")
        st.stop()
else:
    st.error(
        "No CSV found. Put your CSV in the same folder as app.py "
        "or upload it using the sidebar."
    )
    st.info(
        "Supported filenames: "
        + ", ".join(DEFAULT_FILES)
    )
    st.stop()

# ============================================================
# GLOBAL FILTERS
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Global Filters")

platform_options = sorted(df["Platform"].dropna().unique().tolist())
content_options = sorted(df["Content_Type"].dropna().unique().tolist())
region_options = sorted(df["Region"].dropna().unique().tolist())

selected_platforms = st.sidebar.multiselect(
    "Platform", platform_options, default=platform_options
)
selected_content = st.sidebar.multiselect(
    "Content Type", content_options, default=content_options
)
selected_regions = st.sidebar.multiselect(
    "Region", region_options, default=region_options
)

date_min = df["Post_Date"].min().date()
date_max = df["Post_Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = date_min, date_max

filtered_df = df[
    df["Platform"].isin(selected_platforms)
    & df["Content_Type"].isin(selected_content)
    & df["Region"].isin(selected_regions)
    & df["Post_Date"].dt.date.between(start_date, end_date)
].copy()

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">📊 Data-Driven Social Engagement Initiative</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">End-to-end social media analytics dashboard covering all 7 project modules.</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "🏠 Overview",
        "🚀 Virality",
        "💬 Sentiment",
        "🧪 A/B Testing",
        "🎯 Optimization",
        "📈 Growth",
        "🔮 Trends",
    ]
)

# ============================================================
# TAB 1 - OVERVIEW
# ============================================================
with tabs[0]:
    st.header("Content Performance Tracker")

    total_posts = len(filtered_df)
    total_views = filtered_df["Views"].sum()
    total_likes = filtered_df["Likes"].sum()
    total_shares = filtered_df["Shares"].sum()
    total_comments = filtered_df["Comments"].sum()
    avg_engagement = filtered_df["Engagement_Rate"].mean()
    avg_viral = filtered_df["Viral_Coefficient"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Posts", f"{total_posts:,}")
    c2.metric("Total Views", fmt_number(total_views))
    c3.metric("Total Likes", fmt_number(total_likes))
    c4.metric("Total Shares", fmt_number(total_shares))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Comments", fmt_number(total_comments))
    c6.metric("Avg Engagement Rate", f"{avg_engagement:.2f}%")
    c7.metric("Avg Viral Coefficient", f"{avg_viral:.5f}")
    c8.metric("Unique Hashtags", f"{filtered_df['Hashtag'].nunique():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    platform_perf = (
        filtered_df.groupby("Platform", as_index=False)
        .agg(
            Posts=("Post_ID", "count"),
            Views=("Views", "sum"),
            Likes=("Likes", "sum"),
            Shares=("Shares", "sum"),
            Comments=("Comments", "sum"),
            Avg_Engagement_Rate=("Engagement_Rate", "mean"),
            Avg_Viral_Coefficient=("Viral_Coefficient", "mean"),
        )
        .sort_values("Views", ascending=False)
    )

    with col1:
        fig = px.bar(
            platform_perf,
            x="Platform",
            y="Avg_Engagement_Rate",
            title="Average Engagement Rate by Platform",
            text_auto=".2f",
        )
        fig.update_yaxes(title="Engagement Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        content_perf = (
            filtered_df.groupby("Content_Type", as_index=False)
            .agg(Avg_Engagement_Rate=("Engagement_Rate", "mean"))
            .sort_values("Avg_Engagement_Rate", ascending=False)
        )
        fig = px.bar(
            content_perf,
            x="Content_Type",
            y="Avg_Engagement_Rate",
            title="Average Engagement by Content Type",
            text_auto=".2f",
        )
        fig.update_yaxes(title="Engagement Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    monthly = (
        filtered_df.groupby("Year_Month", as_index=False)
        .agg(
            Views=("Views", "sum"),
            Likes=("Likes", "sum"),
            Shares=("Shares", "sum"),
            Comments=("Comments", "sum"),
            Avg_Engagement=("Engagement_Rate", "mean"),
        )
        .sort_values("Year_Month")
    )

    fig = px.line(
        monthly,
        x="Year_Month",
        y="Views",
        markers=True,
        title="Monthly Views Trend",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Performing Posts")
    top_posts = filtered_df.nlargest(
        10, "Viral_Coefficient"
    )[
        [
            "Post_ID",
            "Post_Date",
            "Platform",
            "Hashtag",
            "Content_Type",
            "Views",
            "Likes",
            "Shares",
            "Comments",
            "Engagement_Rate",
            "Viral_Coefficient",
        ]
    ].copy()
    st.dataframe(top_posts, use_container_width=True, hide_index=True)

# ============================================================
# TAB 2 - VIRALITY
# ============================================================
with tabs[1]:
    st.header("🚀 Virality Prediction Engine")

    st.markdown(
        """
        <div class="section-note">
        The supplied dataset has Shares but no Saves. Therefore the dashboard uses
        Shares as the strongest available high-value action, followed by Comments and Likes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    viral_level = (
        filtered_df.groupby("Engagement_Level", as_index=False)
        .agg(
            Posts=("Post_ID", "count"),
            Avg_Viral_Coefficient=("Viral_Coefficient", "mean"),
            Median_Viral_Coefficient=("Viral_Coefficient", "median"),
            Max_Viral_Coefficient=("Viral_Coefficient", "max"),
        )
        .sort_values("Avg_Viral_Coefficient", ascending=False)
    )

    with col1:
        fig = px.bar(
            viral_level,
            x="Engagement_Level",
            y="Avg_Viral_Coefficient",
            title="Average Viral Coefficient by Engagement Level",
            text_auto=".5f",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hashtag_viral = (
            filtered_df.groupby("Hashtag", as_index=False)
            .agg(
                Avg_Viral_Coefficient=("Viral_Coefficient", "mean"),
                Avg_Engagement=("Engagement_Rate", "mean"),
                Total_Shares=("Shares", "sum"),
                Posts=("Post_ID", "count"),
            )
            .sort_values("Avg_Viral_Coefficient", ascending=False)
            .head(15)
        )
        fig = px.bar(
            hashtag_viral,
            x="Avg_Viral_Coefficient",
            y="Hashtag",
            orientation="h",
            title="Top Hashtags by Viral Coefficient",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Random Forest Engagement-Level Prediction")

    model_features = [
        "Platform",
        "Hashtag",
        "Content_Type",
        "Region",
        "Views",
        "Likes",
        "Shares",
        "Comments",
    ]

    model_df = filtered_df[model_features + ["Engagement_Level"]].dropna().copy()

    if model_df["Engagement_Level"].nunique() >= 2 and len(model_df) >= 20:
        X = model_df[model_features]
        y = model_df["Engagement_Level"]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )

            cat_features = ["Platform", "Hashtag", "Content_Type", "Region"]
            num_features = ["Views", "Likes", "Shares", "Comments"]

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        cat_features,
                    ),
                    ("num", "passthrough", num_features),
                ]
            )

            model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=42,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            st.success(f"Model test accuracy: {accuracy:.2%}")

            prediction_table = pd.DataFrame(
                {
                    "Actual": y_test.values,
                    "Predicted": predictions,
                }
            )
            st.dataframe(
                prediction_table.head(20),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.warning(f"Prediction model could not be trained on the current filter: {exc}")
    else:
        st.info(
            "The current filter does not contain enough records/classes to train the "
            "Random Forest model. Select more data."
        )

    st.subheader("Top Viral Posts")
    st.dataframe(
        filtered_df.nlargest(15, "Viral_Coefficient")[
            [
                "Post_ID",
                "Platform",
                "Hashtag",
                "Content_Type",
                "Views",
                "Shares",
                "Comments",
                "Engagement_Rate",
                "Viral_Coefficient",
                "Engagement_Level",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# TAB 3 - SENTIMENT
# ============================================================
with tabs[2]:
    st.header("💬 Audience Sentiment Analyzer")

    has_comment_text = any(
        any(word in str(col).lower() for word in ["comment_text", "comment text", "message", "text"])
        for col in filtered_df.columns
    )

    if has_comment_text:
        text_candidates = [
            c
            for c in filtered_df.columns
            if any(
                word in str(c).lower()
                for word in ["comment_text", "comment text", "message", "text"]
            )
        ]
        st.success(
            "Comment-text column detected: " + ", ".join(map(str, text_candidates))
        )
        st.info(
            "For a full NLP sentiment workflow, provide a file containing actual "
            "comment text. The supplied project CSV normally contains Comments as a numeric count."
        )
    else:
        st.warning(
            "Individual comment text is not available in this dataset."
        )
        st.markdown(
            """
            Your main CSV stores **Comments as a numeric count**, not actual comment
            sentences. Therefore, this dashboard does **not** invent sentiment labels.

            The correct project behavior is:
            - Main social-media CSV → engagement/comment-count analytics
            - Separate comment-text CSV → NLTK + TextBlob/SpaCy sentiment analysis

            Expected future comment file columns:
            `Post_ID` (optional) and `Comment_Text` (required).
            """
        )

    # Useful audience proxy even without comment text.
    st.subheader("Audience Interaction Proxy")

    sentiment_proxy = filtered_df.assign(
        Comment_Rate=(
            filtered_df["Comments"]
            / filtered_df["Views"].replace(0, np.nan)
            * 100
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            sentiment_proxy,
            x="Comment_Rate",
            nbins=30,
            title="Distribution of Comment Rate",
        )
        fig.update_xaxes(title="Comments / Views (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        proxy_by_content = (
            sentiment_proxy.groupby("Content_Type", as_index=False)
            .agg(
                Avg_Comment_Rate=("Comment_Rate", "mean"),
                Avg_Engagement=("Engagement_Rate", "mean"),
            )
            .sort_values("Avg_Comment_Rate", ascending=False)
        )
        fig = px.bar(
            proxy_by_content,
            x="Content_Type",
            y="Avg_Comment_Rate",
            title="Average Comment Rate by Content Type",
            text_auto=".2f",
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4 - A/B TESTING
# ============================================================
with tabs[3]:
    st.header("🧪 A/B Testing Framework")

    st.info(
        "The supplied dataset does not contain randomized experiment IDs or an explicit "
        "A/B assignment. This page therefore performs an observational two-group comparison, "
        "not a causal randomized A/B test."
    )

    content_types = sorted(
        filtered_df["Content_Type"].dropna().unique().tolist()
    )

    if len(content_types) >= 2:
        col1, col2 = st.columns(2)
        group_a = col1.selectbox("Group A", content_types, index=0)
        group_b_options = [x for x in content_types if x != group_a]
        group_b = col2.selectbox("Group B", group_b_options, index=0)

        metric = st.selectbox(
            "Metric",
            ["Engagement_Rate", "Views", "Likes", "Shares", "Comments", "Viral_Coefficient"],
        )

        a = filtered_df.loc[
            filtered_df["Content_Type"] == group_a, metric
        ].dropna()
        b = filtered_df.loc[
            filtered_df["Content_Type"] == group_b, metric
        ].dropna()

        if len(a) >= 2 and len(b) >= 2:
            t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

            mean_a = a.mean()
            mean_b = b.mean()
            difference = mean_a - mean_b

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Group A Mean", f"{mean_a:.4f}")
            r2.metric("Group B Mean", f"{mean_b:.4f}")
            r3.metric("Difference A - B", f"{difference:.4f}")
            r4.metric("P-value", f"{p_value:.6f}")

            if p_value < 0.05:
                st.success(
                    "Statistically significant difference at α = 0.05."
                )
            else:
                st.warning(
                    "No statistically significant difference at α = 0.05."
                )

            comparison = pd.DataFrame(
                {
                    "Group": [group_a, group_b],
                    "Posts": [len(a), len(b)],
                    "Mean": [mean_a, mean_b],
                    "Std Dev": [a.std(), b.std()],
                }
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)

            fig = px.box(
                filtered_df[
                    filtered_df["Content_Type"].isin([group_a, group_b])
                ],
                x="Content_Type",
                y=metric,
                points=False,
                title=f"{metric} Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Both selected groups need at least two observations.")
    else:
        st.info("At least two content types are required for comparison.")

# ============================================================
# TAB 5 - OPTIMIZATION
# ============================================================
with tabs[4]:
    st.header("🎯 Engagement Optimization Recommender")

    strategy = (
        filtered_df.groupby(
            ["Platform", "Hashtag", "Content_Type"], as_index=False
        )
        .agg(
            Posts=("Post_ID", "count"),
            Avg_Engagement_Rate=("Engagement_Rate", "mean"),
            Avg_Viral_Coefficient=("Viral_Coefficient", "mean"),
            Avg_Views=("Views", "mean"),
            Total_Shares=("Shares", "sum"),
        )
    )

    min_posts = st.slider(
        "Minimum posts required for a strategy",
        min_value=1,
        max_value=max(1, min(20, int(strategy["Posts"].max()))),
        value=min(1, int(strategy["Posts"].max())),
    )

    strategy_filtered = strategy[strategy["Posts"] >= min_posts].copy()

    if strategy_filtered.empty:
        st.warning("No strategy combination meets the selected minimum.")
    else:
        strategy_filtered["Optimization_Score"] = (
            strategy_filtered["Avg_Engagement_Rate"].rank(pct=True) * 0.5
            + strategy_filtered["Avg_Viral_Coefficient"].rank(pct=True) * 0.3
            + strategy_filtered["Avg_Views"].rank(pct=True) * 0.2
        )

        strategy_filtered = strategy_filtered.sort_values(
            ["Optimization_Score", "Avg_Engagement_Rate"],
            ascending=False,
        )

        best = strategy_filtered.iloc[0]

        st.success(
            f"Recommended strategy: **{best['Platform']} + "
            f"{best['Hashtag']} + {best['Content_Type']}**"
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Avg Engagement", f"{best['Avg_Engagement_Rate']:.2f}%")
        r2.metric("Avg Viral Coefficient", f"{best['Avg_Viral_Coefficient']:.5f}")
        r3.metric("Avg Views", fmt_number(best["Avg_Views"]))
        r4.metric("Posts", f"{int(best['Posts']):,}")

        st.subheader("Top Recommended Strategies")
        st.dataframe(
            strategy_filtered.head(15),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            strategy_filtered.head(10).sort_values(
                "Avg_Engagement_Rate"
            ),
            x="Avg_Engagement_Rate",
            y="Hashtag",
            color="Content_Type",
            orientation="h",
            title="Top Strategies by Average Engagement",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Best Platform / Hashtag / Content Type")
    b1, b2, b3 = st.columns(3)

    best_platform = (
        filtered_df.groupby("Platform")["Engagement_Rate"].mean().idxmax()
    )
    best_hashtag = (
        filtered_df.groupby("Hashtag")["Engagement_Rate"].mean().idxmax()
    )
    best_content = (
        filtered_df.groupby("Content_Type")["Engagement_Rate"].mean().idxmax()
    )

    b1.metric("Best Platform", str(best_platform))
    b2.metric("Best Hashtag", str(best_hashtag))
    b3.metric("Best Content Type", str(best_content))

# ============================================================
# TAB 6 - GROWTH
# ============================================================
with tabs[5]:
    st.header("📈 Growth Visualization Dashboard")

    monthly = (
        filtered_df.groupby("Year_Month", as_index=False)
        .agg(
            Views=("Views", "sum"),
            Likes=("Likes", "sum"),
            Shares=("Shares", "sum"),
            Comments=("Comments", "sum"),
            Avg_Engagement=("Engagement_Rate", "mean"),
            Avg_Viral=("Viral_Coefficient", "mean"),
        )
        .sort_values("Year_Month")
    )

    metric = st.selectbox(
        "Growth metric",
        ["Views", "Likes", "Shares", "Comments", "Avg_Engagement", "Avg_Viral"],
    )

    fig = px.line(
        monthly,
        x="Year_Month",
        y=metric,
        markers=True,
        title=f"Monthly {metric}",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        platform_views = (
            filtered_df.groupby("Platform", as_index=False)["Views"]
            .sum()
            .sort_values("Views", ascending=False)
        )
        fig = px.bar(
            platform_views,
            x="Platform",
            y="Views",
            title="Total Views by Platform",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        region_perf = (
            filtered_df.groupby("Region", as_index=False)
            .agg(Avg_Engagement=("Engagement_Rate", "mean"))
            .sort_values("Avg_Engagement", ascending=False)
        )
        fig = px.bar(
            region_perf,
            x="Region",
            y="Avg_Engagement",
            title="Average Engagement by Region",
            text_auto=".2f",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Data")
    st.dataframe(monthly, use_container_width=True, hide_index=True)

# ============================================================
# TAB 7 - TRENDS
# ============================================================
with tabs[6]:
    st.header("🔮 Trend Forecasting Module")

    st.info(
        "This forecast uses historical hashtag activity. It is not external Google Trends/API data."
    )

    monthly_hashtag = (
        filtered_df.groupby(["Hashtag", "Year_Month"], as_index=False)
        .agg(Post_Count=("Post_ID", "count"))
        .sort_values(["Hashtag", "Year_Month"])
    )

    trend_rows = []

    for hashtag, group in monthly_hashtag.groupby("Hashtag"):
        group = group.sort_values("Year_Month").copy()

        if len(group) < 2:
            continue

        x = np.arange(len(group), dtype=float)
        y = group["Post_Count"].to_numpy(dtype=float)

        try:
            slope, intercept = np.polyfit(x, y, 1)
            forecast = max(0.0, intercept + slope * len(group))
            latest = float(y[-1])
            growth = forecast - latest

            if latest != 0:
                growth_pct = growth / latest * 100
            else:
                growth_pct = np.nan

            if slope > 0.05:
                status = "Rising"
            elif slope < -0.05:
                status = "Declining"
            else:
                status = "Stable"

            trend_rows.append(
                {
                    "Hashtag": hashtag,
                    "Slope": slope,
                    "Latest_Month_Posts": latest,
                    "Forecast_Next_Month_Posts": forecast,
                    "Growth_Forecast": growth,
                    "Growth_Percent": growth_pct,
                    "Trend": status,
                }
            )
        except Exception:
            continue

    trend_df = pd.DataFrame(trend_rows)

    if trend_df.empty:
        st.warning(
            "Not enough monthly history to calculate hashtag trends. "
            "Select a wider date range."
        )
    else:
        rising = int((trend_df["Trend"] == "Rising").sum())
        stable = int((trend_df["Trend"] == "Stable").sum())
        declining = int((trend_df["Trend"] == "Declining").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Rising Hashtags", rising)
        c2.metric("Stable Hashtags", stable)
        c3.metric("Declining Hashtags", declining)

        top_trends = trend_df.sort_values(
            "Growth_Forecast", ascending=False
        ).head(15)

        fig = px.bar(
            top_trends.sort_values("Growth_Forecast"),
            x="Growth_Forecast",
            y="Hashtag",
            color="Trend",
            orientation="h",
            title="Forecasted Change in Monthly Hashtag Volume",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trend Forecast Table")
        st.dataframe(
            trend_df.sort_values(
                "Growth_Forecast", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

        selected_hashtag = st.selectbox(
            "Inspect hashtag",
            sorted(trend_df["Hashtag"].dropna().unique().tolist()),
        )

        history = monthly_hashtag[
            monthly_hashtag["Hashtag"] == selected_hashtag
        ].sort_values("Year_Month")

        fig = px.line(
            history,
            x="Year_Month",
            y="Post_Count",
            markers=True,
            title=f"Historical Monthly Posts: {selected_hashtag}",
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FOOTER / DOWNLOAD
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption(f"Records after filters: {len(filtered_df):,}")

csv_download = filtered_df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    "⬇️ Download Filtered Data",
    data=csv_download,
    file_name="social_engagement_filtered.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "Data-Driven Social Engagement Initiative • Python • Pandas • NumPy • "
    "Scikit-learn • SciPy • Plotly • Streamlit"
)

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
from data_analysis import get_feature_vectors, get_different_types, filter_non_played, plot_clusters, train_test_model
from data_analysis import get_shap_values, plot_shap_example, normalize_features, plot_global_shap, plot_tree_classifier
from data_analysis import determine_number_of_clusters
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.metrics import normalized_mutual_info_score
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans #type: ignore


st.title("JOINclusion Dashboard")
data = st.file_uploader("Choose a JSON file", type="json")

if data:
    data = json.load(data)
    data = filter_non_played(data) 
    interaction_vectors, full_vectors, usernames, feature_names = get_feature_vectors(data)

    unique_backgrounds, unique_languages, unique_ethnicities = get_different_types(data)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.metric("Total Users", len(usernames))
    with col2:
        st.metric("Unique Backgrounds", len(unique_backgrounds))
    with col3:
        st.metric("Languages", len(unique_languages))
    with col4:
        st.metric("Ethnicities", len(unique_ethnicities))

    df = pd.DataFrame(full_vectors, columns=feature_names)
    df_interactions = pd.DataFrame(interaction_vectors, columns=feature_names[:len(interaction_vectors[0])])
    selected_language = st.selectbox(
        "Select language",
        options=["All"]+unique_languages,
    )


    language_students = []
    for student in data:
        if selected_language == "All":
            language_students.append(student)
        else:
            if student["survey"]['Language'] == selected_language:
                language_students.append(student)

    if 'normalized' not in st.session_state:
        st.session_state.normalized = False

    language_interaction_vectors, language_full_vectors, language_usernames, feature_names = get_feature_vectors(language_students)

    normalized = st.checkbox('Normalize data', value=False)
    if normalized:
        language_interaction_vectors, language_full_vectors = normalize_features(language_interaction_vectors, feature_names), normalize_features(language_full_vectors, feature_names)
    else:
        language_interaction_vectors, language_full_vectors, language_usernames, feature_names = get_feature_vectors(language_students)

    # student_id = 0
    # print(f"{language_usernames[student_id]} has full vector:")
    # for i in range(len(language_interaction_vectors[student_id])):
    #     print(f" - {feature_names[i]}: {language_interaction_vectors[student_id][i]}") # if language_interaction_vectors[student_id][i] != 0 else None


    tabs = st.tabs(["Data Analysis", "Clustering", "Score Estimation"])
    with tabs[0]:
        features_to_show = ["total_score", "num_sessions", "total_time_played", "total_interactions", "total_helps", "age", "migration_age", "adoption", "gender_sex"]
        # features_to_show = feature_names#+["age", "migration_age", "adoption", "gender_sex"]

        # features_to_show = feature_names[:len(interaction_vectors[0])]+["age", "migration_age", "adoption", "gender_sex"]
        language_df = pd.DataFrame(language_full_vectors, columns=feature_names)
        filtered_language_df = language_df[[col for col in features_to_show if col in language_df.columns]]

        st.write(f"{selected_language} ({len(language_df)} users)")
        st.dataframe(language_df)

        selected_feature = st.pills(
            "Select feature to visualize",
            options=features_to_show,
            default="total_score"
        )

        # Distribution plot
        fig = px.histogram(
            language_df,
            x=selected_feature,
            title=f"Distribution of {selected_feature}",
            nbins=30
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title=selected_feature,
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)

        
        col1, col2 = st.columns(2)
        with col1:
            feature_x = st.pills(
                            "Select x axis feature",
                            options=features_to_show,
                            default="total_score",
                            key="featurex"  
                        )        
        with col2:    
            feature_y = st.pills(
                            "Select y axis feature",
                            options=features_to_show,
                            default="total_score",
                            key="featurey"  
                        )        

        fig = px.scatter(
            language_df,
            x=feature_x,
            y=feature_y,
            title=f"Correlation of {feature_x} with {feature_y}",
            trendline='ols'  
        )  
        fig.update_layout(
            showlegend=False,
            xaxis_title=feature_x,
            yaxis_title=feature_y
        )
        st.plotly_chart(fig, use_container_width=True)

        correlation_df = filtered_language_df.copy()
        constant_columns = correlation_df.nunique()[correlation_df.nunique() <= 1].index.tolist()
        correlation_df = correlation_df.drop(columns=constant_columns, errors='ignore')
        # correlation_df = correlation_df[[col for col in feature_names[:len(interaction_vectors[0])] if col in correlation_df.columns]]
        correlation_matrix = correlation_df.corr().round(2)

        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap"
        )
        fig.update_layout(
            width=700, 
            height=700,
            xaxis_tickfont=dict(size=16),
            yaxis_tickfont=dict(size=16),
            font=dict(size=16), 
        )
        st.plotly_chart(fig, use_container_width=True)


    with tabs[1]:

        vectors_dict = {
            "Interaction Vectors": language_interaction_vectors,
            "Full Vectors": language_full_vectors
        }
        selected_label = st.pills(
            "Select Type",
            options=list(vectors_dict.keys()),
            default="Interaction Vectors"
        )
        selected_vectors = vectors_dict[selected_label]



        col1, col2 = st.columns(2)
        with col1:
            n_clusters, kmeans_labels, silhouette_kmeans = determine_number_of_clusters(selected_vectors, streamlit=True)
        with col2:
            _, hierarchical_labels, silhouette_hierarchical = determine_number_of_clusters(selected_vectors, hierarchical=True, streamlit=True)
        silhouette_difference = abs(silhouette_kmeans[n_clusters-2] - silhouette_hierarchical[n_clusters-2])
        nmi = normalized_mutual_info_score(kmeans_labels, hierarchical_labels)
        ari = adjusted_rand_score(kmeans_labels, hierarchical_labels)

        conf_matrix = confusion_matrix(kmeans_labels, hierarchical_labels)

        row_ind, col_ind = linear_sum_assignment(-conf_matrix)
        best_mapping = conf_matrix[row_ind, col_ind].sum()
        overlapping_proportion = best_mapping/len(kmeans_labels)
        minimum_ari = 0.15
        max_silhouette_difference = 0.1
        minimum_nmi = 0.3
        max_clusters = 2
        to_plot_clusters = True
        if silhouette_difference > max_silhouette_difference:
            to_plot_clusters = False
            st.write(f'Clusters are not reliable because the silhouette difference is high ({silhouette_difference:.2f}), indicating uneven cluster quality.')
        if ari < minimum_ari:
            to_plot_clusters = False
            st.write(f'Clusters are not reliable because the Adjusted Rand Index (ARI) is low ({ari:.2f})')
        if nmi < minimum_nmi:
            to_plot_clusters = False
            st.write(f'Clusters are not reliable because the Normalized Mutual Information (NMI) is low ({nmi:.2f})')
        if n_clusters > max_clusters:
            if silhouette_difference <= max_silhouette_difference and silhouette_kmeans[0] > 0.2:
                n_clusters = 2
                model = KMeans(n_clusters=n_clusters)
                model.fit(selected_vectors)
                kmeans_labels = model.labels_
            else:
                to_plot_clusters = False
                st.write(f'There are too many clusters ({n_clusters})')
        if to_plot_clusters:
            st.write(f"The adjusted Rand Index is `{ari:.2f}`")
            st.write(f'This means that the optimal number of clusters is `{n_clusters}`')
            st.write(f'Proportion of same labels: `{overlapping_proportion:.2f}`, Normalized Mutual Information (NMI): `{nmi:.2f}`')
            st.write(f"Students per cluster: {', '.join(f'Cluster {i}: `{count}`' for i, count in enumerate(np.bincount(kmeans_labels)))}")
            plot_clusters(selected_vectors, kmeans_labels, language_usernames, feature_names)
        


            st.write("#### Shap values")
            indices = np.arange(len(selected_vectors))
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                selected_vectors,  
                kmeans_labels,   
                indices,    
                test_size=0.2,
                stratify=kmeans_labels,
                random_state=42 
            )

            test_usernames = [language_usernames[i] for i in test_idx]

            classifier = st.selectbox(
                "Pick a classifier", 
                options=["XGBoost","Decision Trees"],
                key='Classifier',
            )
            use_XGBoost = True
            if classifier == "Decision Trees":
                use_XGBoost = False
            classifier = train_test_model(X_train, X_test, y_train, y_test, use_XGBoost)

            shap_values = get_shap_values(classifier, X_test, n_clusters)
            if use_XGBoost == False:
                plot_tree_classifier(classifier, feature_names, ["Cluster 0", "Cluster 1"])

            if 'selected_user' not in st.session_state or st.session_state.selected_user not in test_usernames:
                st.session_state.selected_user = test_usernames[0]

            student_name = st.selectbox(
                "Pick a user", 
                options=test_usernames,
                key='selected_user'
            )
            test_id = test_usernames.index(student_name)
            student_id = language_usernames.index(student_name) 

            # pred_label = classifier.predict([X_test[test_id]])[0]
            bar_chart = pd.Series(selected_vectors[student_id], index=feature_names[:len(selected_vectors[student_id])])
            non_zero_features = bar_chart[bar_chart != 0]
            fig = px.bar(x=non_zero_features.index, y=non_zero_features.values)
            
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(fig)

            username = language_usernames[student_id]
            plot_shap_example(shap_values, feature_names, test_id, X_test, y_test, username, classifier)

            st.write("### Overall Shap")
            plot_global_shap(shap_values, feature_names)

    with tabs[2]:

        score_indices = [i for i, feature in enumerate(feature_names) if feature in ["total_score","total_score_s1", "total_score_s2"]]
        feature_names_excluding_scores = [feature for i, feature in enumerate(feature_names) if i not in score_indices]

        X = [[vector[i] for i in range(len(vector)) if i not in score_indices] for vector in language_full_vectors]
        y = [vector[0] for vector in language_full_vectors]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        _, usernames_test = train_test_split(language_usernames, test_size=0.3, random_state=None)  
        
        models = [Ridge(), LinearRegression()]
        model = models[0]
        model.fit(X, y)
        y_pred = model.predict(X_test)

        st.metric(label="R² Score", value=f"{model.score(X, y):.3f}")

        fig = px.scatter(x=y_test, y=y_pred,
                        labels={'x': 'Actual Value', 'y': 'Predicted Value'},
                        title='Actual vs Predicted Values',
                        hover_data={'username': usernames_test}) 

        fig.add_shape(type='line', x0=min(y_test), y0=min(y_test),
                    x1=max(y_test), y1=max(y_test),
                    line=dict(color='red', dash='dash'))
        st.plotly_chart(fig)

        coefficients = pd.DataFrame({
        'Feature': feature_names_excluding_scores,
        'Coefficient': model.coef_
        })
        coefficients = coefficients.sort_values('Coefficient', ascending=True)

        fig = go.Figure(go.Bar(
            x=coefficients['Coefficient'],
            y=coefficients['Feature'],
            orientation='h'
        ))
        fig.update_layout(title='Feature Importance',
                        xaxis_title='Coefficient Value',
                        yaxis_title='Features')
        st.plotly_chart(fig)
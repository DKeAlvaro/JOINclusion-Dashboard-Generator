import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, r2_score
import plotly.graph_objects as go
from sklearn.model_selection import KFold
import shap
import pandas as pd
import streamlit as st
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score #type: ignore 
from sklearn.cluster import AgglomerativeClustering

from json import loads

def read_json_data(filename):
    f = open(filename, "r")
    data = loads(f.read())
    f.close()
    return data

def determine_number_of_clusters(data, hierarchical=False, streamlit=False):
    n_clusters = 0
    silhouettes = []
    best_labels = None
    max_num_clusters = 10
    for i in range(2, max_num_clusters):
        model = KMeans(n_clusters=i, n_init=10)
        model_name = "KMeans"
        if hierarchical == True:
            model = AgglomerativeClustering(n_clusters=i)
            model_name = "AgglomerativeClustering"
        model.fit(data)
        labels = model.labels_
        score = silhouette_score(data, labels)
        silhouettes.append(score)
        if score == max(silhouettes):
            best_labels = labels
    n_clusters = silhouettes.index(max(silhouettes)) + 2
    if streamlit:
        fig = px.line(
            x=list(range(2, max_num_clusters)),
            y=silhouettes,
            title=f"Silhouette Scores using {model_name}",
            labels={"x": "Number of Clusters", "y": "Silhouette Score"}
        )
        fig.update_layout()
        st.plotly_chart(fig, use_container_width=True)
    return n_clusters, best_labels, silhouettes


def filter_non_played(data):    
    filtered_students = 0
    filtered_data = []
    for student in data:
        ethnicity = student["survey"]["Ethnicity"]
        if ethnicity == "" or ethnicity == "0":
            student["survey"]["Ethnicity"] = "Ethnicity Not specified"
        tot_score = 0
        for session in student["data"]:
            tot_score += session["scores"]["total_score"]
        if tot_score != 0:
            filtered_data.append(student)
        else:
            filtered_students += 1
    if filtered_students != 0:
        print(f"Filtered students: {filtered_students} out of {len(data)}. (Score of 0)")
    return filtered_data

def get_different_types(students_data):
    unique_backgrounds = []
    unique_languages = []
    unique_ethnicities = []
    for student in students_data:
        background = student["survey"]["MigrantBackground"]
        if background not in unique_backgrounds:
            unique_backgrounds.append(background)
        language = student["survey"]["Language"]
        if language not in unique_languages:
            unique_languages.append(language)

        ethnicity = student["survey"]["Ethnicity"]
        if ethnicity not in unique_ethnicities:
            unique_ethnicities.append(ethnicity)

    return unique_backgrounds, unique_languages, unique_ethnicities

def one_hot_encode(value, unique_values):
    one_hot_vector = [0] * len(unique_values)
    index = unique_values.index(value)
    one_hot_vector[index] = 1
    return one_hot_vector

def get_student_vector(student, unique_backgrounds, unique_languages, unique_ethnicities):

    background = student["survey"]["MigrantBackground"]
    language = student["survey"]["Language"]
    ethnicity = student["survey"]["Ethnicity"]
    age                = int(student["survey"]["Age"]) 
    migration_age      = int(student["survey"]["MigrationAge"]) if student["survey"]["MigrationAge"] != "" else -1
    adoption           = 1 if student["survey"]["Adopted"] == 'Yes' else 0
    gender_sex         = 1 if student["survey"]["Sex"] == 'Boy' else 0

    total_score = 0
    total_time_played = 0
    total_interactions = 0
    total_helps = 0

    total_score_s1 = 0
    total_score_s2 = 0
    total_help_s1 = 0
    total_help_s2 = 0

    total_character_interactions_s1 = 0
    total_character_interactions_s2 = 0
    total_change_scene_interactions_s1 = 0
    total_change_scene_interactions_s2 = 0
    total_movement_interactions_s1 = 0
    total_movement_interactions_s2 = 0

    num_sessions = 0
    
    for session in student["data"]:
        if session["scores"]["total_score"] != 0:
            total_score += session["scores"]["total_score"]/session["scores"]["max_score"]
            total_time_played += session["duration"]
            total_interactions += session["interaction"]["total_interactions"]
            total_helps += session["helps"]["total_help"]

            total_score_s1 += session["scores"]["breakdown"][0]["total_score"]/session["scores"]["breakdown"][0]["max_score"] if session["scores"]["breakdown"][0]["max_score"] != 0 else 0
            total_score_s2 += session["scores"]["breakdown"][1]["total_score"]/session["scores"]["breakdown"][1]["max_score"] if session["scores"]["breakdown"][1]["max_score"] != 0 else 0
            total_help_s1 += session["helps"]["breakdown"][0]["total"]
            total_help_s2 += session["helps"]["breakdown"][1]["total"]

            total_character_interactions_s1 += session["interaction"]["character"]["breakdown"][0]["total"]
            total_character_interactions_s2 += session["interaction"]["character"]["breakdown"][1]["total"]
            total_change_scene_interactions_s1 += session["interaction"]["change_scene"]["breakdown"][0]["total"]
            total_change_scene_interactions_s2 += session["interaction"]["change_scene"]["breakdown"][1]["total"]
            total_movement_interactions_s1 += session["interaction"]["movement"]["breakdown"][0]["total"]
            total_movement_interactions_s2 += session["interaction"]["movement"]["breakdown"][1]["total"]

            num_sessions += 1
    total_score *= 10

    interaction_vector = [total_score, num_sessions, total_time_played, total_interactions, total_helps]
    interaction_vector.extend([total_score_s1, total_score_s2, total_help_s1, total_help_s2])
    interaction_vector.extend([total_character_interactions_s1, total_character_interactions_s2])
    interaction_vector.extend([total_change_scene_interactions_s1, total_change_scene_interactions_s2])
    interaction_vector.extend([total_movement_interactions_s1, total_movement_interactions_s2])

    full_vector = interaction_vector.copy()
    full_vector.extend([age, migration_age, adoption, gender_sex])
    
    background_vector = one_hot_encode(background, unique_backgrounds)
    language_vector = one_hot_encode(language, unique_languages)
    ethnicity_vector = one_hot_encode(ethnicity, unique_ethnicities)
    demographic_vector = background_vector + language_vector + ethnicity_vector
    
    full_vector.extend(demographic_vector)
    return interaction_vector, full_vector

def normalize_features(student_vectors, feature_names):
    features_to_normalize = feature_names[:len(student_vectors[0])]
    arr = np.array(student_vectors)    
    for feature in features_to_normalize:
        feature_idx = feature_names.index(feature)
        column = arr[:, feature_idx]
        min_val = np.nanmin(column) 
        max_val = np.nanmax(column)  
        if min_val != max_val: 
            arr[:, feature_idx] = (arr[:, feature_idx] - min_val) / (max_val - min_val)    
    return arr.tolist()  

def get_feature_vectors(students_data):
    unique_backgrounds, unique_languages, unique_ethnicities = get_different_types(students_data)
    features = ["total_score", "num_sessions", "total_time_played", "total_interactions", "total_helps"]
    features.extend(["total_score_s1", "total_score_s2", "total_help_s1", "total_help_s2"])
    features.extend(["total_character_interactions_s1", "total_character_interactions_s2"])
    features.extend(["total_change_scene_interactions_s1", "total_change_scene_interactions_s2"])
    features.extend(["total_movement_interactions_s1", "total_movement_interactions_s2"])
    features.extend(["age", "migration_age", "adoption", "gender_sex"])
    features.extend(unique_backgrounds)
    features.extend(unique_languages)
    features.extend(unique_ethnicities)

    interaction_vectors = []
    full_vectors = []
    usernames = []
    index = 0
    for student in students_data:
        intr_vector, full_vector = get_student_vector(student, unique_backgrounds, unique_languages, unique_ethnicities)
        interaction_vectors.append(intr_vector)
        full_vectors.append(full_vector)
        usernames.append(student["student"])
        index += 1

    return interaction_vectors, full_vectors, usernames, features

def train_test_model(X_train, X_test, y_train, y_test, XGBoost=False):
    classifier = XGBClassifier() if XGBoost else DecisionTreeClassifier(max_depth=2)   
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    st.write(f"**Classifier F1 Score (weighted)**: {f1:.2f}")
    
    return classifier

def plot_tree_classifier(clf, feature_names, class_names):
    plt.figure(figsize=(20,10))
    plot_tree(clf, 
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=18)

    st.pyplot(plt.gcf())
    plt.clf() 

def get_shap_values(model, X_test, n_classes):
    X_test = np.asarray(X_test, dtype=np.float32)
    explainer = shap.TreeExplainer(model)
    shap_values_list = explainer.shap_values(X_test)
    
    if n_classes == 2:
        if isinstance(shap_values_list, list):
            shap_values = shap_values_list[1]
        else:
            if isinstance(model, DecisionTreeClassifier):
                shap_values = shap_values_list[:, 1]
            else:
                shap_values = shap_values_list
    else:
        if isinstance(model, DecisionTreeClassifier):
            shap_values = np.moveaxis(shap_values_list, 0, -1)
        else:
            shap_values = np.array(shap_values_list)
    
    return np.round(shap_values, decimals=2)

# ------------------------------------------------DASHBOARD METHODS------------------------------------------------------

def plot_shap_example(shap_values, feature_names, sample_idx, X, y, username, classifier):
   
   values = shap_values[sample_idx]
   true_label = y[sample_idx]
   pred_label = classifier.predict([X[sample_idx]])[0]
   
   fig = go.Figure(go.Bar(
       x=values,
       y=feature_names,
       orientation='h'
   ))
   
   fig.update_layout(
       title=f'SHAP Values for {username}<br>True Cluster: {true_label},  Predicted Cluster: {pred_label}',
       xaxis_title='SHAP Value',
       yaxis_title='Features'
   )
   
   st.plotly_chart(fig)

def plot_global_shap(shap_values, feature_names):
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Sort features by importance
    feature_importance = pd.DataFrame(list(zip(feature_names, mean_abs_shap)), 
                                    columns=['feature', 'importance'])
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Average global SHAP Values',
        xaxis_title='SHAP Value',
        yaxis_title='Features',
        height=max(400, len(feature_names) * 20)
    )
    
    st.plotly_chart(fig)

def plot_clusters(vectors, labels, usernames, features, plot_width=800, plot_height=600):
    # Reduce the vectors to 3D using PCA
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)
    data = {
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'z': reduced_vectors[:, 2],
        'label': labels,
        'username': usernames
    }
    
    fig = px.scatter_3d(
        data,
        x='x', y='y', z='z',
        color='label',
        hover_name='username',
        title="3D Cluster Plot",
        labels={"label": "Cluster"},
        hover_data={'x': False, 'y': False, 'z': False, 'username': False, 'label': True},
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ),
        width=plot_width,  
        height=plot_height  
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
def main():
    pilot_two_data = filter_non_played(read_json_data(r'C:\Users\Alvaro\Desktop\JOINCLUSION DATA\pilot_2.json'))
    # pilot_one_data = filter_non_played(read_json_data(r'C:\Users\Alvaro\Desktop\JOINCLUSION DATA\pilot_1.json'))
    data_to_use = pilot_two_data
    full_vectors, full_vectors, usernames, features = get_feature_vectors(data_to_use)

    student_id = 0
    print(f"{usernames[student_id]} has full vector:")
    for i in range(len(full_vectors[student_id])):
        print(f" - {features[i]}: {full_vectors[student_id][i]}") # if full_vectors[student_id][i] != 0 else None

    session_durations = [vector[2] for vector in full_vectors]
    # plt.hist(session_durations, bins=20)
    # plt.show()

if __name__ == "__main__":
    main()
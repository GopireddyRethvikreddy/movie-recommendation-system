from flask import Flask, render_template, request
from data_loader import load_data, create_user_item_matrix
from recommender import Recommender

app = Flask(__name__)

# Load data and initialize model once at startup
ratings_df, movies_df = load_data()
user_item_matrix = create_user_item_matrix(ratings_df)
recommender = Recommender(user_item_matrix)
recommender.compute_similarity()

@app.route('/')
def index():
    return render_template('index.html', movies=movies_df.to_dict('records'))

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form['user_id'])
        recommendations = recommender.get_user_recommendations(user_id)
        
        recommended_movies = []
        if recommendations:
            for movie_id in recommendations:
                movie_info = movies_df[movies_df['MovieID'] == movie_id].iloc[0]
                recommended_movies.append({
                    'Title': movie_info['Title'],
                    'Genres': movie_info['Genres']
                })
        
        return render_template('index.html', 
                             movies=movies_df.to_dict('records'),
                             recommendations=recommended_movies,
                             user_id=user_id)
    except ValueError:
        return render_template('index.html', 
                             movies=movies_df.to_dict('records'),
                             error="Invalid User ID")

if __name__ == '__main__':
    app.run(debug=True)

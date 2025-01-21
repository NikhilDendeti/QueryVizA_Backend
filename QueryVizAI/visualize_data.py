import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
import json

def visualize_data():
    """
    Function to visualize user data as a bar chart including all fields from the dataset.

    Returns:
        dict: A dictionary containing the chart visualization or an error message.
    """
    try:
        # Input data
        data = [
            {'id': 1000, 'first_name': 'Gerti', 'last_name': 'Auckland', 'email': 'gaucklandrr@answers.com', 'gender': 'Female', 'ip_address': '205.225.135.237'},
            {'id': 999, 'first_name': 'Marlane', 'last_name': 'Grunson', 'email': 'mgrunsonrq@prnewswire.com', 'gender': 'Genderqueer', 'ip_address': '50.21.6.98'},
            {'id': 998, 'first_name': 'Trace', 'last_name': 'Woloschinski', 'email': 'twoloschinskirp@ehow.com', 'gender': 'Polygender', 'ip_address': '136.175.31.209'},
            {'id': 997, 'first_name': 'Otto', 'last_name': 'Rodge', 'email': 'orodgero@google.ca', 'gender': 'Male', 'ip_address': '16.13.45.134'},
            {'id': 996, 'first_name': 'Melvyn', 'last_name': 'Wellard', 'email': 'mwellardrn@vk.com', 'gender': 'Non-binary', 'ip_address': '33.98.20.182'},
            {'id': 995, 'first_name': 'Gigi', 'last_name': 'Scawton', 'email': 'gscawtonrm@hud.gov', 'gender': 'Female', 'ip_address': '247.46.9.169'},
            {'id': 994, 'first_name': 'Liva', 'last_name': 'Fynan', 'email': 'lfynanrl@yelp.com', 'gender': 'Female', 'ip_address': '253.35.97.34'},
            {'id': 993, 'first_name': 'Stan', 'last_name': 'Dinan', 'email': 'sdinanrk@nba.com', 'gender': 'Male', 'ip_address': '51.120.248.38'},
            {'id': 992, 'first_name': 'Tandie', 'last_name': 'Feander', 'email': 'tfeanderrj@chronoengine.com', 'gender': 'Female', 'ip_address': '132.169.207.2'},
            {'id': 991, 'first_name': 'Junia', 'last_name': 'Flucker', 'email': 'jfluckerri@independent.co.uk', 'gender': 'Female', 'ip_address': '43.70.52.88'}
        ]

        # Process gender data and include all fields in the hover text
        genders = [entry['gender'] for entry in data]
        counts = {gender: genders.count(gender) for gender in set(genders)}

        # Create hover text with all fields
        hover_text = [
            f"ID: {entry['id']}<br>First Name: {entry['first_name']}<br>Last Name: {entry['last_name']}<br>"
            f"Email: {entry['email']}<br>Gender: {entry['gender']}<br>IP Address: {entry['ip_address']}"
            for entry in data
        ]

        # Generate the bar chart with enhanced UI
        figure = make_subplots(
            rows=1, cols=1,
            subplot_titles=("Gender Distribution with Details"),
        )

        figure.add_trace(
            go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                text=list(counts.values()),
                textposition='auto',
                hovertext=hover_text,
                hoverinfo="text",
                marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'],
            )
        )

        # Update layout for better visuals
        figure.update_layout(
            title="Gender Distribution Bar Chart with Details",
            title_font_size=20,
            xaxis_title="Gender",
            yaxis_title="Count",
            xaxis=dict(
                tickfont=dict(size=14),
                title_font=dict(size=16)
            ),
            yaxis=dict(
                tickfont=dict(size=14),
                title_font=dict(size=16)
            ),
            template="plotly_white",
            plot_bgcolor="#F9F9F9",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        # Show the figure in a browser
        figure.show()

        # Convert the chart to JSON for visualization
        graph_json = json.dumps(figure, cls=PlotlyJSONEncoder)
        return {"success": True, "graph": graph_json}

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

# Example usage
if __name__ == "__main__":
    response = visualize_data()
    print(response)

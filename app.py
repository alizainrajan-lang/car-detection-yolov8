import os
import io
import sys
from flask import Flask, render_template, request, redirect, url_for, send_file
from PIL import Image
from detect_cars_standalone import detect_cars_pure_func

# Passing __name__ directly as string if it is main, or just __name__ var
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure temp dirs exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/processed', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        try:
            # Open image directly from stream
            img = Image.open(file.stream)
            
            # Run detection
            # Pass PIL object
            processed_img, count = detect_cars_pure_func(img)
            
            if processed_img:
                # Save processed image to static folder to serve it
                filename = f"rec_{file.filename}"
                save_path = os.path.join('static', 'processed', filename)
                processed_img.save(save_path)
                
                return render_template('result.html', 
                                       original_filename=file.filename, 
                                       result_image=filename, 
                                       count=count)
            else:
                return "Error processing image", 500
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return f"Error processing request: {e}", 500

if __name__ == '__main__':
    # Fix for some IDLE/Editor environments where __file__ might ideally be used or naming issues
    app.run(debug=True, port=5000, use_reloader=False)

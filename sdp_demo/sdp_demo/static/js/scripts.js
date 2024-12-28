/*!
* Start Bootstrap - Landing Page v6.0.6 (https://startbootstrap.com/theme/landing-page)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-landing-page/blob/master/LICENSE)
*/
// This file is intentionally blank
// Use this file to add JavaScript to your project
function adjustBoxSize(boxId, imageUrl) {
    const box = document.getElementById(boxId);
    const img = new Image();
    img.onload = function () {
        // Set the height based on the image's aspect ratio
        const aspectRatio = this.height / this.width;
        const width = box.offsetWidth; // Get the box's width
        box.style.height = `${width * aspectRatio}px`; // Adjust height dynamically
    };
    img.src = imageUrl; // Load the image
    box.style.backgroundImage = `url('${imageUrl}')`; // Set the background image
}

const fileInput = document.getElementById('autoUploadFile');
const statusDiv = document.getElementById('status');
//const scrolto = document.getElementById('scrolto');
const detection_place = document.getElementById('detection_place');
const detection_paragraph = document.getElementById('detection_paragraph')
const emptied_place = document.getElementById('emptied_place')
const nodes_paragraph = document.getElementById('nodes_paragraph')
const nodes_place = document.getElementById('nodes_place')
const final_place = document.getElementById('final_place')
const final_paragraph = document.getElementById('final_paragraph')
const valueless_netlist = document.getElementById('valueless_netlist')
const netlist = document.getElementById('netlist')
const netlist_place = document.getElementById('netlist_place')
const analysis = document.getElementById('analysis')

fileInput.addEventListener('change', function() {
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        // Show uploading status
        statusDiv.textContent = 'Processing...';

        // Send file via AJAX
        fetch('upload/', { // Replace with your server endpoint
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                return response.text(); // Assuming server returns a success message
            } else {
                throw new Error('Upload failed');
            }
        })
        .then(result => {
            statusDiv.textContent = `Upload successful`;
            console.log(result)
            let jsonObject = JSON.parse(result)

            detection_place.style.backgroundImage = 'url("' + jsonObject.detection_path + '")';
            detection_paragraph.textContent = jsonObject.detection_summary
            //detection_place.style.backgroundSize = 'contain';
            detection_place.scrollIntoView({ behavior: 'smooth', block: 'center' });


            emptied_place.style.backgroundImage = 'url("' + jsonObject.emptied_path + '")';


            nodes_place.style.backgroundImage = 'url("' + jsonObject.contour_path + '")';
            //nodes_paragraph.textContent = jsonObject.contour_text;

            final_place.style.backgroundImage = 'url("' + jsonObject.final_path + '")';
            final_paragraph.textContent = jsonObject.contour_text;

            //valueless_netlist.textContent = jsonObject.valueless_netlist;
            let formattedText = jsonObject.valueless_netlist.replace(/\n/g, '<br>'); // Replace new lines with <br>
            valueless_netlist.innerHTML = formattedText;
            
            let formattedText2 = jsonObject.fixed_netlist.replace(/\n/g, '<br>'); // Replace new lines with <br>
            netlist.innerHTML = formattedText2;
            netlist_place.style.backgroundImage = 'url("' + jsonObject.enum_path + '")';
            
            let formattedText3 = jsonObject.analysis_text.replace(/\n/g, '<br>'); // Replace new lines with <br>
            analysis.innerHTML = formattedText3;
        })
        .catch(error => {
            statusDiv.textContent = `Error: ${error.message}`;
        });
    }
});
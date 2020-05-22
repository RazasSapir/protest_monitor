function plusSlides(n, no) {
  showSlides(slideIndex[no] += n, no);
}

function currentSlide(n, no) {
  showSlides(n, no);
}

function showSlides(n, no) {
  var i;
  var x = document.getElementsByClassName(slideId[no]);
  slideIndex[no] = n
  if (n > x.length) {slideIndex[no] = 1}
  if (n < 1) {slideIndex[no] = x.length}
  for (i = 0; i < x.length; i++) {
     x[i].style.display = "none";
  }
  x[slideIndex[no]-1].style.display = "block";
}

function listToDate(list) {
  var dates = [];
  for (i = 0; i < list.length; i++)
    dates.push(new Date(list[i]));
  return dates;
}

// Open the Modal
function openModal() {
  document.getElementById("myModal").style.display = "block";
}

// Close the Modal
function closeModal() {
  document.getElementById("myModal").style.display = "none";
}

function currentModal(num) {
  document.getElementById("Modal" + num.toString()).style.display = "block";
}
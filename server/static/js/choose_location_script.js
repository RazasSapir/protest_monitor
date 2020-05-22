(function() {
  window.inputNumber = function(el) {
    var min = el.attr('min') || false;
    var max = el.attr('max') || false;
    var els = {};

    els.dec = el.prev();
    els.inc = el.next();

    el.each(function() {
      init($(this));
    });

    function init(el) {
      els.dec.on('click', decrement);
      els.inc.on('click', increment);

      function decrement() {
        var value = el[0].value;
        value--;
        if(!min || value >= min) {
          el[0].value = value;
        }
      }

      function increment() {
        var value = el[0].value;
        value++;
        if(!max || value <= max) {
          el[0].value = value++;
        }
      }
    }
  }
})();

inputNumber($('.input-number'));

var delayTimer;
function querySearch(obj){
    var wait_circle = document.getElementById("wait_circle");
    wait_circle.style.display = 'block';
    clearTimeout(delayTimer);
    delayTimer = setTimeout(function() {
        var value = obj.value;
        $.get("query_location/" + value, function(data, status){
            wait_circle.style.display = 'none';
            var loc_list = JSON.parse(data);
            var select = document.getElementById("select_loc");
            var loc_wrap = document.getElementById("search_results");
            $(select).empty();
            select.style.display = 'block';
            loc_wrap.style.display = 'block';
            for (var i = 0; i < loc_list.length; i++) {
                var option = document.createElement("option");
                option.text = decodeURIComponent(loc_list[i][0]);
                option.value = decodeURIComponent(loc_list[i][1]);
                select.add(option);
            }
        });
    }, 1000);
}

function clearResults(obj){
    $(obj).blur(function()
    {
          if( !this.value ) {
                var wait_circle = document.getElementById("wait_circle");
                wait_circle.style.display = 'none';
                var select = document.getElementById("select_loc");
                var loc_wrap = document.getElementById("search_results");
                $(select).empty();
                select.style.display = 'none';
                loc_wrap.style.display = 'none';
                $(this).parents('p').addClass('warning');
          }
    });
}
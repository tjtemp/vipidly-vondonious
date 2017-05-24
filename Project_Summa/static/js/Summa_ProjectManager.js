/**
 * Created by joo on 17. 2. 12.
 */
$(document).ready(function(){

    $('#datatable').dataTable();

    $('#dattable-keytable').DataTable({
      keys: true
    });

    $('#datatable-responsive').DataTable();

    var $datatable = $('.datatable-checkbox');

    $datatable.dataTable({
      'order': [[ 1, 'desc' ]],
      'columnDefs': [
        { orderable: false, targets: [0] }
      ],
    });
    $datatable.on('draw.dt', function() {
      $('input').iCheck({
        checkboxClass: 'icheckbox_flat-green'
      });
    });

    $('input[type="checkbox"]').on('ifChecked', function(event){

        alert(event.type + ' callback');
//        var table = $(this).closest("table");
//        console.log(typeof(table));
//        console.log(typeof($datatable));
//        console.log(table == $datatable); // false
//
//        console.log($datatable.fnGetNodes());
//        console.log(table.fnGetNodes());
        //$(this).closest("table");

    });

    $('.delete_yes_button').on('click', function(){
        //alert('delete call');
        //var table = $(this).closest('div > .task-table-wrapper')
        var selected_row = $('.btn-task-delete').parent().siblings().children().find('tr.selected');
        var selected_task_pk = selected_row.find('td.task-pk');
        var ajax_delete_differer = $(this).val();
//        console.log(selected_row);
//        alert(selected_row.index());
//        alert(selected_task_pk);
//        alert(ajax_delete_differer);
        var task_pk_array = [];
        for(var i=0;i<selected_task_pk.length;i++){
            console.log(selected_task_pk[i].innerHTML);
            task_pk_array.push(selected_task_pk[i].innerHTML);
        }

        var csrftoken = $.cookie('csrftoken');

        function csrfSafeMethod(method) {
            // these HTTP methods do not require CSRF protection
            return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
        }

        $.ajaxSetup({
            beforeSend: function(xhr, settings) {
                if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                    xhr.setRequestHeader("X-CSRFToken", csrftoken);
                }


 //               alert('ajax before send!');
                // this code moved to index.html

                var table = $('#task-table1').DataTable();

                $('#delete_yes_button1').click( function () {
                    alert('function called!');
                    table.row('.selected').remove().draw( false );
                } );

            }
        });


        $.ajax({
            type: 'POST',
            url: '/projectmanager/',
            data: {'task_pks[]': task_pk_array, 'ajax_differer': ajax_delete_differer},
            success: function(data){
               alert('DB deleted! and ' + data);
            },
        });
    });


});


$('#top-padding').css('padding-top',30)


$(document).ready(function(){
  //var svg = d3.select("svg"),
  var svg = d3.select("#task-graph").append("svg").attr("width",700).attr("height",700),
      width = +svg.attr("width"),
      height = +svg.attr("height");

  var color = d3.scaleOrdinal(d3.schemeCategory20);

  var simulation = d3.forceSimulation()
      .force("link", d3.forceLink().id(function(d) { return d.id; }))
      .force("charge", d3.forceManyBody())
      .force("center", d3.forceCenter(width / 2, height / 2));

  d3.json("/static/graph_temp/miserables.json", function(error, graph) {
    if (error) throw error;
    var pretty = JSON.stringify(graph, null, 2);
    //console.log(pretty);
    var test = {"nodes":[1,2,3],"b":[3,4,5]}
    //console.log(graph["nodes"])
    //console.log(graph["links"])


    var link = svg.append("g")
        .attr("class", "links")
      .selectAll("line")
      .data(graph.links)
      .enter().append("line")
        .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

    var node = svg.append("g")
        .attr("class", "nodes")
      .selectAll("circle")
      .data(graph.nodes)
      .enter().append("circle")
        .attr("r", 5)
        .attr("fill", function(d) { return color(d.group); })
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("title")
        .text(function(d) { return d.id; });

    simulation
        .nodes(graph.nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(graph.links);

    function ticked() {
      link
          .attr("x1", function(d) { return d.source.x; })
          .attr("y1", function(d) { return d.source.y; })
          .attr("x2", function(d) { return d.target.x; })
          .attr("y2", function(d) { return d.target.y; });

      node
          .attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; });
    }
  });

  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
  }

  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
});

$(documnet).ready(function(){
var margin = {top: 20, right: 120, bottom: 20, left: 120},
    width = 960 - margin.right - margin.left,
    height = 800 - margin.top - margin.bottom;

var i = 0,
    duration = 750,
    root;

var tree = d3.layout.tree()
    .size([height, width]);

var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.y, d.x]; });

// original source
// var svg = d3.select("body").append("svg")
//     .attr("width", width + margin.right + margin.left)
//     .attr("height", height + margin.top + margin.bottom)
//   .append("g")
//     .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


// first method
// var svg = d3.select("#category-tree").append("svg").attr("width",700).attr("height",700),
//       width = +svg.attr("width"),
//       height = +svg.attr("height"),
//       g = svg.append("g").attr("transform", "translate(40,0)");


// modified
var svg = d3.select("#category-tree").append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


d3.json("ProjectManagerTree.json", function(error, flare) {
  if (error) throw error;

  root = flare;
  root.x0 = height / 2;
  root.y0 = 0;

  function collapse(d) {
    if (d.children) {
      d._children = d.children;
      d._children.forEach(collapse);
      d.children = null;
    }
  }

  root.children.forEach(collapse);
  update(root);
});

d3.select(self.frameElement).style("height", "800px");

function update(source) {

  // Compute the new tree layout.
  var nodes = tree.nodes(root).reverse(),
      links = tree.links(nodes);

  // Normalize for fixed-depth.
  nodes.forEach(function(d) { d.y = d.depth * 180; });

  // Update the nodes…
  var node = svg.selectAll("g.node")
      .data(nodes, function(d) { return d.id || (d.id = ++i); });

  // Enter any new nodes at the parent's previous position.
  var nodeEnter = node.enter().append("g")
      .attr("class", "node")
      .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
      .on("click", click);

  nodeEnter.append("circle")
      .attr("r", 1e-6)
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

  nodeEnter.append("text")
      .attr("x", function(d) { return d.children || d._children ? -10 : 10; })
      .attr("dy", ".35em")
      .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
      .text(function(d) { return d.name; })
      .style("fill-opacity", 1e-6);

  // Transition nodes to their new position.
  var nodeUpdate = node.transition()
      .duration(duration)
      .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

  nodeUpdate.select("circle")
      .attr("r", 4.5)
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

  nodeUpdate.select("text")
      .style("fill-opacity", 1);

  // Transition exiting nodes to the parent's new position.
  var nodeExit = node.exit().transition()
      .duration(duration)
      .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
      .remove();

  nodeExit.select("circle")
      .attr("r", 1e-6);

  nodeExit.select("text")
      .style("fill-opacity", 1e-6);

  // Update the links…
  var link = svg.selectAll("path.link")
      .data(links, function(d) { return d.target.id; });

  // Enter any new links at the parent's previous position.
  link.enter().insert("path", "g")
      .attr("class", "link")
      .attr("d", function(d) {
        var o = {x: source.x0, y: source.y0};
        return diagonal({source: o, target: o});
      });

  // Transition links to their new position.
  link.transition()
      .duration(duration)
      .attr("d", diagonal);

  // Transition exiting nodes to the parent's new position.
  link.exit().transition()
      .duration(duration)
      .attr("d", function(d) {
        var o = {x: source.x, y: source.y};
        return diagonal({source: o, target: o});
      })
      .remove();

  // Stash the old positions for transition.
  nodes.forEach(function(d) {
    d.x0 = d.x;
    d.y0 = d.y;
  });
}

// Toggle children on click.
function click(d) {
  if (d.children) {
    d._children = d.children;
    d.children = null;
  } else {
    d.children = d._children;
    d._children = null;
  }
  update(d);
}
});
/*
$(document).ready(function(){
  //var svg = d3.select("svg[id=category-tree]"),
  var svg = d3.select("#category-tree").append("svg").attr("width",700).attr("height",700),
      width = +svg.attr("width"),
      height = +svg.attr("height"),
      g = svg.append("g").attr("transform", "translate(40,0)");

  var tree = d3.cluster()
      .size([height, width - 160]);

  var stratify = d3.stratify()
      .parentId(function(d) { return d.id.substring(0, d.id.lastIndexOf(".")); });

  d3.csv("/static/graph_temp/flare.csv", function(error, data) {
    if (error) throw error;
    //console.log(data)
    var root = stratify(data)
        .sort(function(a, b) { return (a.height - b.height) || a.id.localeCompare(b.id); });

    tree(root);

    var link = g.selectAll(".link")
        .data(root.descendants().slice(1))
      .enter().append("path")
        .attr("class", "link")
        .attr("d", function(d) {
          return "M" + d.y + "," + d.x
              + "C" + (d.parent.y + 100) + "," + d.x
              + " " + (d.parent.y + 100) + "," + d.parent.x
              + " " + d.parent.y + "," + d.parent.x;
        });

    var node = g.selectAll(".node")
        .data(root.descendants())
      .enter().append("g")
        .attr("class", function(d) { return "node" + (d.children ? " node--internal" : " node--leaf"); })
        .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; })

    node.append("circle")
        .attr("r", 2.5);

    node.append("text")
        .attr("dy", 3)
        .attr("x", function(d) { return d.children ? -8 : 8; })
        .style("text-anchor", function(d) { return d.children ? "end" : "start"; })
        .text(function(d) { return d.id.substring(d.id.lastIndexOf(".") + 1); });
  });
});

*/
$(document).ready(function(){
    $('#controller-task-post-form').on('submit', function(e){
        e.preventDefault();
        $.ajax({
            url: "/projectmanager/", //this is the submit URL
            type: "POST", //or POST
            data: $('#controller-task-post-form').serialize(),
            csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val(),
            success: function(data){
                 alert('successfully submitted');
            }
        });
    });
});

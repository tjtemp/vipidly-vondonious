/**
 * Created by joo on 17. 2. 12.
 */
    // Controller panel 에 Categories and workspace table 을 선언합니다.
    $(document).ready(function() {
        $('#ctr1-catogory-workspace-table').DataTable();
        $('#project-panel-project-table').DataTable();
    } );

    // Controller panel 에 ctr1-task-table1 을 선언합니다.
    $(document).ready(function() {
        var table = $('#ctr1-task-table1').DataTable( {
            columnDefs: [ {
                orderable: false,
                className: 'select-checkbox',
                targets:   0
            } ],
            select: {
                style:    'multi',
                selector: 'td:first-child'
            },
            order: [[ 1, 'asc' ]]
        } );

    $('.delete_yes_button').on('click', function(){
        // JQuery DataTable api 를 이용하지 않고 task.pk 값을 받아옵니다.
        /*
        var table = $(this).closest('div > .task-table-wrapper')
        var selected_row = $('.btn-task-delete').parent().siblings().children().find('tr.selected');
        var selected_task_pk = selected_row.find('td.task-pk');
        var task_pk_array = [];
        for(var i=0;i<selected_task_pk.length;i++){
            console.log(selected_task_pk[i].innerHTML);
            task_pk_array.push(selected_task_pk[i].innerHTML);
        }
        */

        // DataTable 에서 task.pk 값을 받아옵니다.
        var task_pks = $.map(table.rows('.selected').data(), function (item) {
            return item[1]
        });
        console.log(task_pks);
        //alert(table.rows('.selected').data().length + ' row(s) selected');


        // 선택된 row들을 테이블에서 지웁니다. (ajaxSetup 에 넣으면 실행 안됩니다.)
        table.rows('.selected').remove().draw( false );

        // 여러 table의 ajax call 을 구분해주는 구분자를 넣어 줍니다.
        var ajax_delete_differer = $(this).val();
        //alert('ajax_delete_differer : '+ajax_delete_differer);

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

            }
        });

        // delete_task should come from delete_yes_button name and value but it's not working right now.
        $.ajax({
            type: 'POST',
            url: '/projectmanager/',
            data: {'task_pks[]': task_pks, 'ajax_differer': ajax_delete_differer, 'delete_task': 1},
            success: function(data){
               //alert('DB deleted! and ' + data);
            },
        });
    });





    } );

    // Controller panel 에 ctr1-contribution-ranker-table 을 선언합니다.
    $(document).ready(function() {
        $('#ctr1-contribution-ranker-table').DataTable( {
            scrollY:        "200px",
            scrollCollapse: true,
            paging:         false,
        } );
    } );

    // Category panel 에 collapsible-tree 를 표현합니다.
    // https://bl.ocks.org/mbostock/4339083
    $(document).ready(function(){

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

        var svg = d3.select("div #collapsible-tree").append("svg")
            .attr("width", width + margin.right + margin.left)
            .attr("height", height + margin.top + margin.bottom)
          .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        d3.json("/static/js/ProjectManagerTree.json", function(error, flare) {
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

    // Category panel 에 member-categorization-graph 를 표현합니다.
    // http://codepen.io/shprink/details/emQMEG
    $(document).ready(function(){
        //
        // Generated by the Exaile Playlist Analyzer plugin.
        // (C) 2014 Dustin Spicuzza <dustin@virtualroadside.com>
        //
        // This work is licensed under the Creative Commons Attribution 4.0
        // International License. To view a copy of this license, visit
        // http://creativecommons.org/licenses/by/4.0/.
        //
        // Inspired by http://www.findtheconversation.com/concept-map/
        // Loosely based on http://bl.ocks.org/mbostock/4063550
        //

        var data = [
            ["testtask", ["like", "call response", "dramatic intro", "has breaks", "male vocalist", "silly", "swing"]],
            [150, ["brassy", "like", "calm energy", "female vocalist", "swing", "fun"]],
            [170, ["calm energy", "instrumental", "swing", "like", "happy"]],
            [140, ["has breaks", "male vocalist", "swing", "piano", "banjo", "chill"]],
            [160, ["calm energy", "instrumental", "swing", "like", "interesting"]],
            [140, ["brassy", "like", "energy", "dramatic intro", "male vocalist", "baseball", "swing"]],
            [170, ["instrumental", "interesting", "high energy", "like", "swing"]],
            [140, ["instrumental", "energy", "like", "swing"]],
            [200, ["instrumental", "brassy", "dramatic intro", "like", "swing"]],
            [160, ["male vocalist", "brassy", "swing", "like", "my favorites"]],
            [130, ["like", "interesting", "dramatic intro", "male vocalist", "silly", "swing", "gospel"]],
            [160, ["like", "long intro", "announcer", "energy", "swing", "female vocalist"]],
            [170, ["instrumental", "swing", "bass", "like"]],
            [150, ["like", "interesting", "has breaks", "instrumental", "chunky", "swing", "banjo", "trumpet"]],
            [170, ["like", "has breaks", "male vocalist", "silly", "swing", "banjo"]],
            [190, ["instrumental", "banjo", "swing"]],
            [130, ["instrumental", "brassy", "banjo", "like", "swing"]],
            [160, ["brassy", "like", "energy", "instrumental", "big band", "jam", "swing"]],
            [150, ["like", "male vocalist", "live", "swing", "piano", "banjo", "chill"]],
            [150, ["like", "trick ending", "instrumental", "chunky", "swing", "chill"]],
            [120, ["brassy", "like", "female vocalist", "swing", "chill", "energy buildup"]],
            [150, ["brassy", "like", "interesting", "instrumental", "swing", "piano"]],
            [190, ["brassy", "like", "long intro", "energy", "baseball", "swing", "female vocalist"]],
            [180, ["calm energy", "female vocalist", "live", "like", "swing"]],
            [200, ["banjo", "like", "long intro", "interesting", "energy", "my favorites", "male vocalist", "silly", "swing", "fun", "balboa"]],
            [150, ["brassy", "calm energy", "chunky", "instrumental", "old-timey", "live", "swing"]],
            [160, ["like", "call response", "interesting", "instrumental", "calm energy", "swing"]],
            [180, ["interesting", "swing", "fast", "male vocalist"]],
            [150, ["calm energy", "chunky", "swing", "female vocalist", "like"]],
            [180, ["like", "has breaks", "male vocalist", "chunky", "silly", "swing"]],
            [140, ["instrumental", "brassy", "dramatic intro", "swing", "chill"]],
            [150, ["male vocalist", "trumpet", "like", "swing"]],
            [150, ["instrumental", "energy", "like", "has breaks", "swing"]],
            [180, ["brassy", "like", "energy", "has breaks", "instrumental", "has calm", "swing"]],
            [150, ["female vocalist", "swing"]],
            [170, ["instrumental", "brassy", "energy", "swing"]],
            [170, ["calm energy", "instrumental", "energy", "like", "swing"]],
            [190, ["brassy", "like", "instrumental", "high energy", "swing", "trumpet"]],
            [160, ["male vocalist", "energy", "swing", "old-timey"]],
            [170, ["like", "oldies", "my favorites", "fast", "male vocalist", "high energy", "swing"]]
        ];

        console.log(data);
        // transform the data into a useful representation
        // 1 is inner, 2, is outer

        // need: inner, outer, links
        //
        // inner:
        // links: { inner: outer: }


        var outer = d3.map();
        var inner = [];
        var links = [];

        var outerId = [0];

        data.forEach(function(d){

            if (d == null)
                return;

            i = { id: 'i' + inner.length, name: d[0], related_links: [] };
            i.related_nodes = [i.id];
            inner.push(i);

            if (!Array.isArray(d[1]))
                d[1] = [d[1]];

            d[1].forEach(function(d1){

                o = outer.get(d1);

                if (o == null)
                {
                    o = { name: d1,	id: 'o' + outerId[0], related_links: [] };
                    o.related_nodes = [o.id];
                    outerId[0] = outerId[0] + 1;

                    outer.set(d1, o);
                }

                // create the links
                l = { id: 'l-' + i.id + '-' + o.id, inner: i, outer: o }
                links.push(l);

                // and the relationships
                i.related_nodes.push(o.id);
                i.related_links.push(l.id);
                o.related_nodes.push(i.id);
                o.related_links.push(l.id);
            });
        });

        data = {
            inner: inner,
            outer: outer.values(),
            links: links
        }

        // sort the data -- TODO: have multiple sort options
        outer = data.outer;
        data.outer = Array(outer.length);


        var i1 = 0;
        var i2 = outer.length - 1;

        for (var i = 0; i < data.outer.length; ++i)
        {
            if (i % 2 == 1)
                data.outer[i2--] = outer[i];
            else
                data.outer[i1++] = outer[i];
        }

        console.log(data.outer.reduce(function(a,b) { return a + b.related_links.length; }, 0) / data.outer.length);


        // from d3 colorbrewer:
        // This product includes color specifications and designs developed by Cynthia Brewer (http://colorbrewer.org/).
        var colors = ["#a50026","#d73027","#f46d43","#fdae61","#fee090","#ffffbf","#e0f3f8","#abd9e9","#74add1","#4575b4","#313695"]
        var color = d3.scale.linear()
            .domain([60, 220])
            .range([colors.length-1, 0])
            .clamp(true);

        var diameter = 960;
        var rect_width = 40;
        var rect_height = 14;

        var link_width = "1px";

        var il = data.inner.length;
        var ol = data.outer.length;

        var inner_y = d3.scale.linear()
            .domain([0, il])
            .range([-(il * rect_height)/2, (il * rect_height)/2]);

        mid = (data.outer.length/2.0)
        var outer_x = d3.scale.linear()
            .domain([0, mid, mid, data.outer.length])
            .range([15, 170, 190 ,355]);

        var outer_y = d3.scale.linear()
            .domain([0, data.outer.length])
            .range([0, diameter / 2 - 120]);


        // setup positioning
        data.outer = data.outer.map(function(d, i) {
            d.x = outer_x(i);
            d.y = diameter/3;
            return d;
        });

        data.inner = data.inner.map(function(d, i) {
            d.x = -(rect_width / 2);
            d.y = inner_y(i);
            return d;
        });


        function get_color(name)
        {
            var c = Math.round(color(name));
            if (isNaN(c))
                return '#dddddd';	// fallback color

            return colors[c];
        }

        // Can't just use d3.svg.diagonal because one edge is in normal space, the
        // other edge is in radial space. Since we can't just ask d3 to do projection
        // of a single point, do it ourselves the same way d3 would do it.


        function projectX(x)
        {
            return ((x - 90) / 180 * Math.PI) - (Math.PI/2);
        }




        var diagonal = d3.svg.diagonal()
            .source(function(d) { return {"x": d.outer.y * Math.cos(projectX(d.outer.x)),
                                          "y": -d.outer.y * Math.sin(projectX(d.outer.x))}; })
            .target(function(d) { return {"x": d.inner.y + rect_height/2,
                                          "y": d.outer.x > 180 ? d.inner.x : d.inner.x + rect_width}; })
            .projection(function(d) { return [d.y, d.x]; });

        var svg = d3.select("div #member-categorization-graph").append("svg")
            .attr("width", diameter)
            .attr("height", diameter)
          .append("g")
            .attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");


        // links
        var link = svg.append('g').attr('class', 'links').selectAll(".link")
            .data(data.links)
          .enter().append('path')
            .attr('class', 'link')
            .attr('id', function(d) { return d.id })
            .attr("d", diagonal)
            .attr('stroke', function(d) { return get_color(d.inner.name); })
            .attr('stroke-width', link_width);

        // outer nodes

        var onode = svg.append('g').selectAll(".outer_node")
            .data(data.outer)
          .enter().append("g")
            .attr("class", "outer_node")
            .attr("transform", function(d) { return "rotate(" + (d.x - 90) + ")translate(" + d.y + ")"; })
            .on("mouseover", mouseover)
            .on("mouseout", mouseout);

        onode.append("circle")
            .attr('id', function(d) { return d.id })
            .attr("r", 4.5);

        onode.append("circle")
            .attr('r', 20)
            .attr('visibility', 'hidden');

        onode.append("text")
            .attr('id', function(d) { return d.id + '-txt'; })
            .attr("dy", ".31em")
            .attr("text-anchor", function(d) { return d.x < 180 ? "start" : "end"; })
            .attr("transform", function(d) { return d.x < 180 ? "translate(8)" : "rotate(180)translate(-8)"; })
            .text(function(d) { return d.name; });

        // inner nodes

        var inode = svg.append('g').selectAll(".inner_node")
            .data(data.inner)
          .enter().append("g")
            .attr("class", "inner_node")
            .attr("transform", function(d, i) { return "translate(" + d.x + "," + d.y + ")"})
            .on("mouseover", mouseover)
            .on("mouseout", mouseout);

        inode.append('rect')
            .attr('width', rect_width)
            .attr('height', rect_height)
            .attr('id', function(d) { return d.id; })
            .attr('fill', function(d) { return get_color(d.name); });

        inode.append("text")
            .attr('id', function(d) { return d.id + '-txt'; })
            .attr('text-anchor', 'middle')
            .attr("transform", "translate(" + rect_width/2 + ", " + rect_height * .75 + ")")
            .text(function(d) { return d.name; });

        // need to specify x/y/etc

        d3.select(self.frameElement).style("height", diameter - 150 + "px");

        function mouseover(d)
        {
            // bring to front
            d3.selectAll('.links .link').sort(function(a, b){ return d.related_links.indexOf(a.id); });

            for (var i = 0; i < d.related_nodes.length; i++)
            {
                d3.select('#' + d.related_nodes[i]).classed('highlight', true);
                d3.select('#' + d.related_nodes[i] + '-txt').attr("font-weight", 'bold');
            }

            for (var i = 0; i < d.related_links.length; i++)
                d3.select('#' + d.related_links[i]).attr('stroke-width', '5px');
        }

        function mouseout(d)
        {
            for (var i = 0; i < d.related_nodes.length; i++)
            {
                d3.select('#' + d.related_nodes[i]).classed('highlight', false);
                d3.select('#' + d.related_nodes[i] + '-txt').attr("font-weight", 'normal');
            }

            for (var i = 0; i < d.related_links.length; i++)
                d3.select('#' + d.related_links[i]).attr('stroke-width', link_width);
        }
    });

    // Category panel 에 search-able-architecture-graph 를 표현합니다.

    // Workspace panel 에 resource-pie-graph 를 표현합니다.
    $(document).ready(function(){
       Highcharts.chart('resource-pie-graph', {
            chart: {
                plotBackgroundColor: null,
                plotBorderWidth: null,
                plotShadow: false,
                type: 'pie'
            },
            title: {
                text: 'Browser market shares January, 2015 to May, 2015'
            },
            tooltip: {
                pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
            },
            plotOptions: {
                pie: {
                    allowPointSelect: true,
                    cursor: 'pointer',
                    dataLabels: {
                        enabled: true,
                        format: '<b>{point.name}</b>: {point.percentage:.1f} %',
                        style: {
                            color: (Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black'
                        }
                    }
                }
            },
            series: [{
                name: 'Brands',
                colorByPoint: true,
                data: [{
                    name: 'Microsoft Internet Explorer',
                    y: 56.33
                }, {
                    name: 'Chrome',
                    y: 24.03,
                    sliced: true,
                    selected: true
                }, {
                    name: 'Firefox',
                    y: 10.38
                }, {
                    name: 'Safari',
                    y: 4.77
                }, {
                    name: 'Opera',
                    y: 0.91
                }, {
                    name: 'Proprietary or Undetectable',
                    y: 0.2
                }]
            }]
        });
    });

    // Workspace panel 에서 project strength 를 spider map 으로 보여줍니다.
    $(document).ready(function(){
        Highcharts.chart('project-strength-spider-map', {

            chart: {
                polar: true
            },

            title: {
                text: 'Highcharts Polar Chart'
            },

            pane: {
                startAngle: 0,
                endAngle: 360
            },

            xAxis: {
                tickInterval: 45,
                min: 0,
                max: 360,
                labels: {
                    formatter: function () {
                        return this.value + '°';
                    }
                }
            },

            yAxis: {
                min: 0
            },

            plotOptions: {
                series: {
                    pointStart: 0,
                    pointInterval: 45
                },
                column: {
                    pointPadding: 0,
                    groupPadding: 0
                }
            },

            series: [{
                type: 'column',
                name: 'Column',
                data: [8, 7, 6, 5, 4, 3, 2, 1],
                pointPlacement: 'between'
            }, {
                type: 'line',
                name: 'Line',
                data: [1, 2, 3, 4, 5, 6, 7, 8]
            }, {
                type: 'area',
                name: 'Area',
                data: [1, 8, 2, 7, 3, 6, 4, 5]
            }]
        });
    });

    // Workspace panel 에서 task-try-open-date 를 보여줍니다.
    $(document).ready(function(){

        Highcharts.chart('task-try-open-date-duration', {

            chart: {
                type: 'columnrange',
                inverted: true
            },

            title: {
                text: 'task try open date duration'
            },

            subtitle: {
                text: 'Observed in Vik i Sogn, Norway'
            },

            xAxis: {
                categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            },

            yAxis: {
                title: {
                    text: 'Temperature ( °C )'
                }
            },

            tooltip: {
                valueSuffix: '°C'
            },

            plotOptions: {
                columnrange: {
                    dataLabels: {
                        enabled: true,
                        formatter: function () {
                            return this.y + '°C';
                        }
                    }
                }
            },

            legend: {
                enabled: false
            },

            series: [{
                name: 'Temperatures',
                data: [
                    [-9.7, 9.4],
                    [-8.7, 6.5],
                    [-3.5, 9.4],
                    [-1.4, 19.9],
                    [0.0, 22.6],
                    [2.9, 29.5],
                    [9.2, 30.7],
                    [7.3, 26.5],
                    [4.4, 18.0],
                    [-3.1, 11.4],
                    [-5.2, 10.4],
                    [-13.5, 9.8]
                ]
            }]

        });

    });

    // Controller panel 에서 view category modal 로부터 cateo .. 보류중..
    $(document).ready(function(){
        var ajax_differer = 2;

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

            }
        });


        $.ajax({
            type: 'POST',
            url: '/projectmanager/',
            data: {'ajax_differer': ajax_differer},
            success: function(data){
               //alert('DB deleted! and ' + data);
            },
        });
    });
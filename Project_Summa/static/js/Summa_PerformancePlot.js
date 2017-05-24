/**
 * Created by joo on 17. 5. 20.
 */
$(document).ready(function(){
    function draw_plot(graph_select_index){
        switch(graph_select_index){
            case 0:
                $(function () {
                    Highcharts.chart('container', {
                        chart: {
                            type: 'scatter',
                            zoomType: 'xy'
                        },
                        title: {
                            text: 'Height Versus Weight of 507 Individuals by Gender'
                        },
                        subtitle: {
                            text: 'Source: Heinz  2003'
                        },
                        xAxis: {
                            title: {
                                enabled: true,
                                text: 'Height (cm)'
                            },
                            startOnTick: true,
                            endOnTick: true,
                            showLastLabel: true
                        },
                        yAxis: {
                            title: {
                                text: 'Weight (kg)'
                            }
                        },
                        legend: {
                            layout: 'vertical',
                            align: 'left',
                            verticalAlign: 'top',
                            x: 100,
                            y: 70,
                            floating: true,
                            backgroundColor: (Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF',
                            borderWidth: 1
                        },
                        plotOptions: {
                            scatter: {
                                marker: {
                                    radius: 5,
                                    states: {
                                        hover: {
                                            enabled: true,
                                            lineColor: 'rgb(100,100,100)'
                                        }
                                    }
                                },
                                states: {
                                    hover: {
                                        marker: {
                                            enabled: false
                                        }
                                    }
                                },
                                tooltip: {
                                    headerFormat: '<b>{series.name}</b><br>',
                                    pointFormat: '{point.x} cm, {point.y} kg'
                                }
                            }
                        },
                        series: [{
                            name: 'Female',
                            color: 'rgba(223, 83, 83, .5)',
                            data: [[161.2, 51.6], [167.5, 59.0], [159.5, 49.2], [157.0, 63.0], [155.8, 53.6],
                                [170.0, 59.0], [159.1, 47.6], [166.0, 69.8], [176.2, 66.8], [160.2, 75.2],
                                [172.5, 55.2], [170.9, 54.2], [172.9, 62.5], [153.4, 42.0], [160.0, 50.0],]
                        }, {
                            name: 'Male',
                            color: 'rgba(119, 152, 191, .5)',
                            data: [[174.0, 65.6], [175.3, 71.8], [193.5, 80.7], [186.5, 72.6], [187.2, 78.8],
                                [181.5, 74.8], [184.0, 86.4], [184.5, 78.4], [175.0, 62.0], [184.0, 81.6],]
                        }]
                    });
                });
                break;
            case 1:
                $(function () {
                    $.getJSON('https://www.highcharts.com/samples/data/jsonp.php?filename=usdeur.json&callback=?', function (data) {
                        var detailChart;

                        $(document).ready(function () {

                            // create the detail chart
                            function createDetail(masterChart) {

                                // prepare the detail chart
                                var detailData = [],
                                    detailStart = data[0][0];

                                $.each(masterChart.series[0].data, function () {
                                    if (this.x >= detailStart) {
                                        detailData.push(this.y);
                                    }
                                });

                                // create a detail chart referenced by a global variable
                                detailChart = Highcharts.chart('detail-container', {
                                    chart: {
                                        marginBottom: 120,
                                        reflow: false,
                                        marginLeft: 50,
                                        marginRight: 20,
                                        style: {
                                            position: 'absolute'
                                        }
                                    },
                                    credits: {
                                        enabled: false
                                    },
                                    title: {
                                        text: 'Historical USD to EUR Exchange Rate'
                                    },
                                    subtitle: {
                                        text: 'Select an area by dragging across the lower chart'
                                    },
                                    xAxis: {
                                        type: 'datetime'
                                    },
                                    yAxis: {
                                        title: {
                                            text: null
                                        },
                                        maxZoom: 0.1
                                    },
                                    tooltip: {
                                        formatter: function () {
                                            var point = this.points[0];
                                            return '<b>' + point.series.name + '</b><br/>' + Highcharts.dateFormat('%A %B %e %Y', this.x) + ':<br/>' +
                                                '1 USD = ' + Highcharts.numberFormat(point.y, 2) + ' EUR';
                                        },
                                        shared: true
                                    },
                                    legend: {
                                        enabled: false
                                    },
                                    plotOptions: {
                                        series: {
                                            marker: {
                                                enabled: false,
                                                states: {
                                                    hover: {
                                                        enabled: true,
                                                        radius: 3
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    series: [{
                                        name: 'USD to EUR',
                                        pointStart: detailStart,
                                        pointInterval: 24 * 3600 * 1000,
                                        data: detailData
                                    }],

                                    exporting: {
                                        enabled: false
                                    }

                                }); // return chart
                            }

                            // create the master chart
                            function createMaster() {
                                Highcharts.chart('master-container', {
                                    chart: {
                                        reflow: false,
                                        borderWidth: 0,
                                        backgroundColor: null,
                                        marginLeft: 50,
                                        marginRight: 20,
                                        zoomType: 'x',
                                        events: {

                                            // listen to the selection event on the master chart to update the
                                            // extremes of the detail chart
                                            selection: function (event) {
                                                var extremesObject = event.xAxis[0],
                                                    min = extremesObject.min,
                                                    max = extremesObject.max,
                                                    detailData = [],
                                                    xAxis = this.xAxis[0];

                                                // reverse engineer the last part of the data
                                                $.each(this.series[0].data, function () {
                                                    if (this.x > min && this.x < max) {
                                                        detailData.push([this.x, this.y]);
                                                    }
                                                });

                                                // move the plot bands to reflect the new detail span
                                                xAxis.removePlotBand('mask-before');
                                                xAxis.addPlotBand({
                                                    id: 'mask-before',
                                                    from: data[0][0],
                                                    to: min,
                                                    color: 'rgba(0, 0, 0, 0.2)'
                                                });

                                                xAxis.removePlotBand('mask-after');
                                                xAxis.addPlotBand({
                                                    id: 'mask-after',
                                                    from: max,
                                                    to: data[data.length - 1][0],
                                                    color: 'rgba(0, 0, 0, 0.2)'
                                                });


                                                detailChart.series[0].setData(detailData);

                                                return false;
                                            }
                                        }
                                    },
                                    title: {
                                        text: null
                                    },
                                    xAxis: {
                                        type: 'datetime',
                                        showLastTickLabel: true,
                                        maxZoom: 14 * 24 * 3600000, // fourteen days
                                        plotBands: [{
                                            id: 'mask-before',
                                            from: data[0][0],
                                            to: data[data.length - 1][0],
                                            color: 'rgba(0, 0, 0, 0.2)'
                                        }],
                                        title: {
                                            text: null
                                        }
                                    },
                                    yAxis: {
                                        gridLineWidth: 0,
                                        labels: {
                                            enabled: false
                                        },
                                        title: {
                                            text: null
                                        },
                                        min: 0.6,
                                        showFirstLabel: false
                                    },
                                    tooltip: {
                                        formatter: function () {
                                            return false;
                                        }
                                    },
                                    legend: {
                                        enabled: false
                                    },
                                    credits: {
                                        enabled: false
                                    },
                                    plotOptions: {
                                        series: {
                                            fillColor: {
                                                linearGradient: [0, 0, 0, 70],
                                                stops: [
                                                    [0, Highcharts.getOptions().colors[0]],
                                                    [1, 'rgba(255,255,255,0)']
                                                ]
                                            },
                                            lineWidth: 1,
                                            marker: {
                                                enabled: false
                                            },
                                            shadow: false,
                                            states: {
                                                hover: {
                                                    lineWidth: 1
                                                }
                                            },
                                            enableMouseTracking: false
                                        }
                                    },

                                    series: [{
                                        type: 'area',
                                        name: 'USD to EUR',
                                        pointInterval: 24 * 3600 * 1000,
                                        pointStart: data[0][0],
                                        data: data
                                    }],

                                    exporting: {
                                        enabled: false
                                    }

                                }, function (masterChart) {
                                    createDetail(masterChart);
                                }); // return chart instance
                            }

                            // make the container smaller and add a second container for the master chart
                            var $container = $('#container')
                                .css('position', 'relative');

                            $('<div id="detail-container">')
                                .appendTo($container);

                            $('<div id="master-container">')
                                .css({
                                    position: 'absolute',
                                    top: 300,
                                    height: 100,
                                    width: '100%'
                                })
                                .appendTo($container);

                            // create master and in its callback, create the detail chart
                            createMaster();
                        });
                    });
                });
                break;
            case 2:
                $(function () {
                    Highcharts.chart('container', {
                        chart: {
                            type: 'column'
                        },
                        title: {
                            text: 'Stacked column chart'
                        },
                        xAxis: {
                            categories: ['Apples', 'Oranges', 'Pears', 'Grapes', 'Bananas']
                        },
                        yAxis: {
                            min: 0,
                            title: {
                                text: 'Total fruit consumption'
                            },
                            stackLabels: {
                                enabled: true,
                                style: {
                                    fontWeight: 'bold',
                                    color: (Highcharts.theme && Highcharts.theme.textColor) || 'gray'
                                }
                            }
                        },
                        legend: {
                            align: 'right',
                            x: -30,
                            verticalAlign: 'top',
                            y: 25,
                            floating: true,
                            backgroundColor: (Highcharts.theme && Highcharts.theme.background2) || 'white',
                            borderColor: '#CCC',
                            borderWidth: 1,
                            shadow: false
                        },
                        tooltip: {
                            headerFormat: '<b>{point.x}</b><br/>',
                            pointFormat: '{series.name}: {point.y}<br/>Total: {point.stackTotal}'
                        },
                        plotOptions: {
                            column: {
                                stacking: 'normal',
                                dataLabels: {
                                    enabled: true,
                                    color: (Highcharts.theme && Highcharts.theme.dataLabelsColor) || 'white'
                                }
                            }
                        },
                        series: [{
                            name: 'John',
                            data: [5, 3, 4, 7, 2]
                        }, {
                            name: 'Jane',
                            data: [2, 2, 3, 2, 1]
                        }, {
                            name: 'Joe',
                            data: [3, 4, 4, 2, 5]
                        }]
                    });
                });
                break;
            case 3:
                //alert('hi');
                break;
            case 4:
                $(function () {
                    var colors = Highcharts.getOptions().colors,
                        categories = ['MSIE', 'Firefox', 'Chrome', 'Safari', 'Opera'],
                        data = [{
                            y: 56.33,
                            color: colors[0],
                            drilldown: {
                                name: 'MSIE versions',
                                categories: ['MSIE 6.0', 'MSIE 7.0', 'MSIE 8.0', 'MSIE 9.0', 'MSIE 10.0', 'MSIE 11.0'],
                                data: [1.06, 0.5, 17.2, 8.11, 5.33, 24.13],
                                color: colors[0]
                            }
                        }, {
                            y: 10.38,
                            color: colors[1],
                            drilldown: {
                                name: 'Firefox versions',
                                categories: ['Firefox v31', 'Firefox v32', 'Firefox v33', 'Firefox v35', 'Firefox v36', 'Firefox v37', 'Firefox v38'],
                                data: [0.33, 0.15, 0.22, 1.27, 2.76, 2.32, 2.31, 1.02],
                                color: colors[1]
                            }
                        }, {
                            y: 24.03,
                            color: colors[2],
                            drilldown: {
                                name: 'Chrome versions',
                                categories: ['Chrome v30.0', 'Chrome v31.0', 'Chrome v32.0', 'Chrome v33.0', 'Chrome v34.0',
                                    'Chrome v35.0', 'Chrome v36.0', 'Chrome v37.0', 'Chrome v38.0', 'Chrome v39.0', 'Chrome v40.0', 'Chrome v41.0', 'Chrome v42.0', 'Chrome v43.0'
                                    ],
                                data: [0.14, 1.24, 0.55, 0.19, 0.14, 0.85, 2.53, 0.38, 0.6, 2.96, 5, 4.32, 3.68, 1.45],
                                color: colors[2]
                            }
                        }, {
                            y: 4.77,
                            color: colors[3],
                            drilldown: {
                                name: 'Safari versions',
                                categories: ['Safari v5.0', 'Safari v5.1', 'Safari v6.1', 'Safari v6.2', 'Safari v7.0', 'Safari v7.1', 'Safari v8.0'],
                                data: [0.3, 0.42, 0.29, 0.17, 0.26, 0.77, 2.56],
                                color: colors[3]
                            }
                        }, {
                            y: 0.91,
                            color: colors[4],
                            drilldown: {
                                name: 'Opera versions',
                                categories: ['Opera v12.x', 'Opera v27', 'Opera v28', 'Opera v29'],
                                data: [0.34, 0.17, 0.24, 0.16],
                                color: colors[4]
                            }
                        }, {
                            y: 0.2,
                            color: colors[5],
                            drilldown: {
                                name: 'Proprietary or Undetectable',
                                categories: [],
                                data: [],
                                color: colors[5]
                            }
                        }],
                        browserData = [],
                        versionsData = [],
                        i,
                        j,
                        dataLen = data.length,
                        drillDataLen,
                        brightness;


                    // Build the data arrays
                    for (i = 0; i < dataLen; i += 1) {

                        // add browser data
                        browserData.push({
                            name: categories[i],
                            y: data[i].y,
                            color: data[i].color
                        });

                        // add version data
                        drillDataLen = data[i].drilldown.data.length;
                        for (j = 0; j < drillDataLen; j += 1) {
                            brightness = 0.2 - (j / drillDataLen) / 5;
                            versionsData.push({
                                name: data[i].drilldown.categories[j],
                                y: data[i].drilldown.data[j],
                                color: Highcharts.Color(data[i].color).brighten(brightness).get()
                            });
                        }
                    }

                    // Create the chart
                    Highcharts.chart('container', {
                        chart: {
                            type: 'pie'
                        },
                        title: {
                            text: 'Browser market share, January, 2015 to May, 2015'
                        },
                        subtitle: {
                            text: 'Source: <a href="http://netmarketshare.com/">netmarketshare.com</a>'
                        },
                        yAxis: {
                            title: {
                                text: 'Total percent market share'
                            }
                        },
                        plotOptions: {
                            pie: {
                                shadow: false,
                                center: ['50%', '50%']
                            }
                        },
                        tooltip: {
                            valueSuffix: '%'
                        },
                        series: [{
                            name: 'Browsers',
                            data: browserData,
                            size: '60%',
                            dataLabels: {
                                formatter: function () {
                                    return this.y > 5 ? this.point.name : null;
                                },
                                color: '#ffffff',
                                distance: -30
                            }
                        }, {
                            name: 'Versions',
                            data: versionsData,
                            size: '80%',
                            innerSize: '60%',
                            dataLabels: {
                                formatter: function () {
                                    // display only if larger than 1
                                    return this.y > 1 ? '<b>' + this.point.name + ':</b> ' + this.y + '%' : null;
                                }
                            }
                        }]
                    });
                });
                break;
            default:

        }

    }

    // {0:'scatter', 1:'line', 2:'bar', 3:'area', 4:'pie'}
    $('#g0').on('click', function(){
        draw_plot(0)
    });
    $('#g1').on('click', function(){
        draw_plot(1)
    });
    $('#g2').on('click', function(){
        draw_plot(2)
    });
    $('#g3').on('click', function(){
        draw_plot(3)
    });
    $('#g4').on('click', function(){
        draw_plot(4)
    });

});



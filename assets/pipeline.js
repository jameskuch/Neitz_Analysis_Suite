/* Analysis Pipelines canvas interactions (companion to viewer.py's node-graph editor).
 *
 * Dash renders the NODES (so their param fields stay real Dash components); this file draws the
 * CONNECTION curves and handles NODE DRAGGING + CLICK-TO-CONNECT. Node/port pixel positions are
 * read straight from the DOM (offset math), so nothing here needs to know the node geometry.
 *
 *   drawPipeConnections(graph) — (re)draw the SVG curves from graph.connections. Called by a Dash
 *                                clientside callback after every node render, and locally on drag.
 *   dragging  — pointerdown on a .pnode-header moves the node live, redraws curves, and on release
 *               writes "node|x|y|nonce" to #pipe-drag-sink (a hidden dcc.Input) so Dash persists it.
 *   connect   — click an OUTPUT port to arm it, then click an INPUT port to wire them: writes
 *               "fromN|fromP|toN|toP|nonce" to #pipe-connect-sink (Dash validates + stores it).
 */
(function () {
  var SVGNS = "http://www.w3.org/2000/svg";
  var nonce = 0;
  var armed = null;   // {node, port, el} of a clicked output port awaiting a target input

  function setDashInput(id, value) {
    var el = document.getElementById(id);
    if (!el) return;
    var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(el, value);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }

  // center of a port dot in #pipe-canvas coordinates (offset math survives scroll + zoom)
  function portCenter(node, portEl) {
    return {
      x: node.offsetLeft + portEl.offsetLeft + portEl.offsetWidth / 2,
      y: node.offsetTop + portEl.offsetTop + portEl.offsetHeight / 2,
    };
  }

  function findPort(nodeId, portName, io) {
    return document.querySelector(
      '.pport[data-node="' + nodeId + '"][data-port="' + portName + '"][data-io="' + io + '"]'
    );
  }

  window.drawPipeConnections = function (graph) {
    var host = document.getElementById("pipe-conn");
    var canvas = document.getElementById("pipe-canvas");
    if (!host || !canvas) return;
    var conns = (graph && graph.connections) || [];
    var svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("width", canvas.scrollWidth || canvas.clientWidth);
    svg.setAttribute("height", canvas.scrollHeight || canvas.clientHeight);
    svg.style.position = "absolute";
    svg.style.top = "0";
    svg.style.left = "0";
    conns.forEach(function (c) {
      var oEl = findPort(c.from_node, c.from_port, "out");
      var iEl = findPort(c.to_node, c.to_port, "in");
      if (!oEl || !iEl) return;
      var oNode = oEl.closest(".pnode"), iNode = iEl.closest(".pnode");
      if (!oNode || !iNode) return;
      var a = portCenter(oNode, oEl), b = portCenter(iNode, iEl);
      var dx = Math.max(40, Math.abs(b.x - a.x) / 2);
      var path = document.createElementNS(SVGNS, "path");
      path.setAttribute("d", "M " + a.x + " " + a.y + " C " + (a.x + dx) + " " + a.y +
                             ", " + (b.x - dx) + " " + b.y + ", " + b.x + " " + b.y);
      path.setAttribute("class", "pipe-conn-path");
      path.setAttribute("stroke", oEl.getAttribute("data-ptype") ?
                        (portColor(oEl.getAttribute("data-ptype"))) : "#7a8296");
      svg.appendChild(path);
    });
    host.innerHTML = "";
    host.appendChild(svg);
  };

  function portColor(ptype) {
    return ({ recordings: "#6fb0ff", spikes: "#c78cff", result: "#ffcf6f", outputs: "#9fe0b0" }
            )[ptype] || "#7a8296";
  }

  function clearArmed() {
    if (armed && armed.el) armed.el.classList.remove("armed");
    armed = null;
  }

  // ---- click-to-connect: output port arms, input port completes ----
  document.addEventListener("click", function (e) {
    var port = e.target.closest && e.target.closest(".pport");
    if (!port) {
      if (e.target.closest && e.target.closest("#pipe-canvas-wrap") && !e.target.closest(".pnode"))
        clearArmed();  // click empty canvas -> disarm
      return;
    }
    e.stopPropagation();
    var io = port.getAttribute("data-io");
    var node = port.getAttribute("data-node"), name = port.getAttribute("data-port");
    if (io === "out") {
      clearArmed();
      armed = { node: node, port: name, el: port };
      port.classList.add("armed");
    } else if (io === "in" && armed) {
      if (armed.node !== node) {
        setDashInput("pipe-connect-sink",
          armed.node + "|" + armed.port + "|" + node + "|" + name + "|" + (nonce++));
      }
      clearArmed();
    }
  });

  // ---- node dragging (pointer events, delegated so it survives Dash re-renders) ----
  var drag = null;   // {node, startX, startY, origLeft, origTop, moved}
  document.addEventListener("pointerdown", function (e) {
    var header = e.target.closest && e.target.closest(".pnode-header");
    if (!header || (e.target.closest && e.target.closest(".pnode-del"))) return;
    var node = header.closest(".pnode");
    if (!node) return;
    drag = {
      node: node,
      id: node.getAttribute("data-node"),
      startX: e.clientX, startY: e.clientY,
      origLeft: node.offsetLeft, origTop: node.offsetTop,
      moved: false,
    };
    node.style.zIndex = 10;
    e.preventDefault();
  });
  document.addEventListener("pointermove", function (e) {
    if (!drag) return;
    var nx = Math.max(0, drag.origLeft + (e.clientX - drag.startX));
    var ny = Math.max(0, drag.origTop + (e.clientY - drag.startY));
    drag.node.style.left = nx + "px";
    drag.node.style.top = ny + "px";
    drag.lastX = nx; drag.lastY = ny; drag.moved = true;
    if (window._pipeGraph) window.drawPipeConnections(window._pipeGraph);
  });
  document.addEventListener("pointerup", function () {
    if (!drag) return;
    drag.node.style.zIndex = "";
    if (drag.moved) {
      setDashInput("pipe-drag-sink",
        drag.id + "|" + Math.round(drag.lastX) + "|" + Math.round(drag.lastY) + "|" + (nonce++));
    }
    drag = null;
  });

  // stash the latest graph so a live drag can redraw connections without a server round-trip
  var _origDraw = window.drawPipeConnections;
  window.drawPipeConnections = function (graph) {
    if (graph) window._pipeGraph = graph;
    _origDraw(graph || window._pipeGraph);
  };
})();

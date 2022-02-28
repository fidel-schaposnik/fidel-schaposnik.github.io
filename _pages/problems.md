---
layout: page
permalink: /problems/
title: problems
description:
nav: false
---

<script>
function filterNames() {
  // Declare variables
  var input, filter, table, tr, td, i, txtValue;
  input = document.getElementById("nameSearch");
  filter = input.value.toUpperCase();
  table = document.getElementById("problemList");
  tr = table.getElementsByTagName("tr");

  // Loop through all table rows, and hide those who don't match the search query
  for (i = 0; i < tr.length; i++) {
    td = tr[i].getElementsByTagName("td")[1];
    if (td) {
      txtValue = td.textContent || td.innerText;
      if (txtValue.toUpperCase().indexOf(filter) > -1) {
        tr[i].style.display = "";
      } else {
        tr[i].style.display = "none";
      }
    }
  }
}
</script>

There are currently {{ site.data.problem_db.total_problems}} problems in the database!

<input type="text" id="nameSearch" onkeyup="filterNames()" placeholder="Search for names.." style="width:100%;padding:10px">

<table id="problemList">
  <tr class="header">
	<th>Judge</th>
    <th>Problem name</th>
    <th>Statement</th>
	<th>Solution</th>
  </tr>

  {%- for problem in site.data.problem_db.problems -%}
  <tr>
	<td>
	  {%- if problem.judge == "LiveArchive"-%}
	     <img src="/assets/img/LiveArchive.png">
	  {%- endif -%}
	</td>
    <td>
	  <a href="https://icpcarchive.ecs.baylor.edu/index.php?option=com_onlinejudge&page=show_problem&problem={{ problem.problem_id }}">{{ problem.problem_name }}</a>
	  {%- for tag in problem.tags -%}
	  <br><abbr class="badge badge-warning">{{ tag }}</abbr>
	  {%- endfor -%}
	</td>
	<td><a href="https://github.com/fidel-schaposnik/icpc-solutions/raw/master/{{ problem.directory | uri_escape}}/{{ problem.statement | uri_escape}}" class="btn btn-sm z-depth-0" role="button">PDF</a></td>
	<td><a href="https://github.com/fidel-schaposnik/icpc-solutions/raw/master/{{ problem.directory | uri_escape}}/{{ problem.code | uri_escape}}" class="btn btn-sm z-depth-0" role="button">CODE</a></td>
  </tr>
  {%- endfor -%}
  
</table>